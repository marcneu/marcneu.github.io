---
layout: post
title: "The DSP Inference Bug That Broke Our Neural Network"
---

## Background

High-Level Synthesis (HLS) has been around for years, and vendors often market it as the solution to all our hardware design challenges. Seasoned FPGA engineers know better, but HLS tools have nonetheless found a foothold in research environments—where delivery speed often takes priority over design quality, and complex problems must be tackled with limited resources.

At Belle II, we use AMD Vitis HLS to generate a Graph Neural Network dataflow accelerator targeting ultra-low-latency, high-throughput applications. Neural network deployments demand continuous improvement, so we update our models frequently. Over the past month, I discovered some surprising issues with HLS tool reliability that I want to share.

## Problem Statement

Dense layers are a staple of any neural network implementation. We use a simple template to implement a common dense layer consisting of a matrix-vector multiplication, vector accumulation, and a comparison for the ReLU activation function. Our template is similar to those provided in HLS frameworks such as FINN or hls4ml, but simplified for our use cases:

```c++
#include <array>
#include "hls_stream.h"

using std::array;

template <typename input_t,
          typename output_t,
          typename weight_t,
          typename bias_t,
          typename accum_t,
          int F_IN,
          int F_OUT,
          int II>
void dense(hls::stream<array<input_t,F_IN>> &inputStream,
           hls::stream<array<output_t,F_OUT>> &outputStream,
           weight_t weights[F_IN * F_OUT],
           bias_t biases[F_OUT]) {
    #pragma HLS ARRAY_PARTITION variable=biases complete
    accum_t mult[F_IN * F_OUT];
    #pragma HLS ARRAY_PARTITION variable=mult complete
    accum_t acc[F_OUT];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    for(int ii = 0; ii < II; ii++) {
        #pragma HLS PIPELINE II=1 style=flp rewind
        array<input_t,F_IN> input;
        inputStream >> input;

        for (int ii = 0; ii < F_IN; ii++) {
            for (int jj = 0; jj < F_OUT; jj++) {
                int index = ii * F_OUT + jj;
                mult[index] = input[ii] * weights[index];
            }
        }

        for (int iacc = 0; iacc < F_OUT; iacc++) {
            acc[iacc] = static_cast<accum_t>(biases[iacc]);
            }

        for (int ii = 0; ii < F_IN; ii++) {
            for (int jj = 0; jj < F_OUT; jj++) {
                int index = ii * F_OUT + jj;
                acc[jj] += mult[index];
            }
        }

        array<output_t,F_OUT> result;

        for (int i = 0; i < F_OUT; i++) {
            if(acc[i] > 0)
                result[i] = static_cast<output_t>(acc[i]);
            else
                result[i] = static_cast<output_t>(0.0);        
            }

        outputStream << result;
    }
}
```

Instantiating this template is straightforward:

```c++
static const int II = 1;
static const int INPUT_WIDTH = 32;
static const int OUTPUT_WIDTH = 32;

typedef ap_fixed<8,3>  input_t;
typedef ap_fixed<8,1>  weight_t;
typedef ap_fixed<8,1>  biases_t;
typedef ap_fixed<22,9> accum_t;
typedef ap_fixed<8,2>  output_t;

static weight_t weights[INPUT_WIDTH * OUTPUT_WIDTH] = {0., 0., -0.2734375, 0.3046875, 0., 0., -0.1328125, 0.0859375, 0.3359375, 0., 0., -0.0859375, 0.140625, 0.2421875, -0.1171875, -0.0390625, 0.2265625, 0., 0.0234375, 0., 0., 0., -0.390625, -0.046875, 0.375, -0.1875, 0., 0.0078125, 0., 0., -0.328125, 0.203125, 0.9921875, 0.9921875, 0., 0.9921875, -0.2421875, 0.9921875, -0.0625, 0., 0., 0.078125, 0.1640625, 0., -0.8203125, 0.9921875, -0.09375, 0., 0., 0., 0.0703125, 0.03125, 0.015625, 0.1875, -0.203125, 0., 0., -0.578125, 0., -0.1328125, 0.4375, 0., 0., -0.140625, 0., 0., 0., 0.203125, 0., -0.109375, 0., 0.2265625, 0., -0.1875, 0., -0.28125, 0., 0., 0.2578125, 0.25, -0.2421875, 0.0078125, 0., 0.453125, -0.1875, 0., 0., 0.0234375, 0.203125, 0., 0.6640625, 0.0234375, 0., 0., -0.2421875, 0., 0., 0.03125, 0., -0.046875, 0., -0.03125, 0., -0.1953125, 0., -0.4453125, -0.25, 0., 0., 0., 0., 0., 0., 0.2734375, -0.1640625, 0., 0.625, 0., 0., -0.1171875, 0., -0.28125, 0.0625, -0.28125, 0.2734375, 0., -0.3203125, 0.3984375, 0.25, -0.3203125, -0.2109375, 0.2734375, -0.1875, -0.15625, 0.46875, 0., 0., 0.5, -0.0390625, 0.484375, 0., 0., 0., 0.2890625, 0.28125, 0., -0.15625, 0., 0., 0.296875, -0.390625, 0., 0., 0.3515625, 0.1171875, -0.609375, 0., 0.0625, 0., 0.328125, 0.0546875, 0., 0.1875, 0.109375, -0.09375, -0.2578125, -0.3046875, 0.21875, 0., 0.3203125, 0.1328125, 0., 0., 0., 0.40625, -0.3125, 0., 0., 0., 0., 0., 0., 0.2734375, 0., 0., -0.0859375, 0.4921875, 0.453125, 0.625, 0., 0.140625, 0., 0.140625, 0., -0.1640625, 0.0546875, 0.3359375, -0.140625, 0., -0.3828125, 0.375, 0.2109375, 0., 0., 0., 0., 0., 0., 0.375, -0.234375, 0., -0.3828125, 0.3203125, 0.3359375, -0.375, 0., -0.1015625, 0.2421875, 0., 0.1328125, 0.2734375, 0.15625, 0., 0.4765625, 0.515625, 0.1484375, 0., 0.359375, -0.0546875, 0.9921875, 0., 0.5234375, 0., 0.1796875, 0., -0.0859375, 0.125, 0.7421875, -0.2265625, 0., 0.375, 0., 0., 0., 0., -0.0703125, -0.3828125, 0., 0.5078125, 0.0546875, 0.140625, 0.0390625, 0., 0., 0.2578125, 0.140625, 0., 0.0390625, 0.40625, 0.1953125, 0.2109375, -0.2421875, 0., 0.359375, -0.1484375, 0.1875, -0.375, 0., 0.28125, -0.2109375, 0.15625, 0., 0., 0., 0., -0.21875, -0.25, 0.2109375, 0., 0., -0.3828125, 0., 0., 0., -0.3828125, -0.109375, 0.4375, 0.734375, 0.9921875, 0.9921875, 0.078125, 0.9921875, -0.375, 0.9921875, 0.078125, 0., -0.3359375, 0.1328125, 0., 0.1015625, -0.9921875, 0.9921875, 0.09375, 0.078125, 0., -0.9921875, -0.1171875, -0.140625, 0.1875, 0.0625, 0., 0., -0.2265625, -0.359375, -0.0390625, -0.015625, -0.2890625, -0.1484375, 0.46875, 0.03125, 0., -0.1953125, 0., 0., 0., -0.28125, 0., 0., -0.3671875, 0., 0.0546875, 0., -0.1875, 0., 0.375, 0., -0.1640625, 0., -0.1640625, 0.203125, -0.4140625, 0., 0.4921875, 0.15625, 0., 0., -0.0546875, 0., 0., -0.5625, 0.15625, 0.2109375, 0., -0.265625, 0., 0., 0., 0., 0.2578125, 0.2421875, -0.265625, 0., 0., 0., 0.015625, 0.015625, 0., 0.421875, 0.34375, 0., 0., 0.3671875, 0.296875, 0., 0.171875, 0., 0.109375, 0., 0., -0.3671875, 0.21875, 0., 0.2109375, -0.109375, 0., 0., 0.2890625, -0.34375, 0., -0.2109375, 0., -0.3828125, -0.2421875, 0., 0.1796875, 0.2578125, 0.578125, 0., 0.125, 0., -0.3515625, 0.2890625, -0.5, 0., 0.1875, -0.2890625, -0.1328125, 0.625, 0.265625, 0., 0., 0., -0.203125, 0., 0.203125, -0.3671875, 0.1484375, 0., 0., 0.203125, -0.0390625, 0., 0., 0., 0.1796875, 0.25, 0.0390625, -0.1015625, 0., -0.03125, 0., 0., 0., 0.484375, -0.046875, 0.40625, 0., -0.1484375, 0., 0.1875, -0.25, 0., -0.3984375, -0.0234375, 0., 0., 0., 0., -0.046875, 0., 0.1640625, 0., 0.3046875, 0.203125, 0., -0.25, 0.40625, 0., -0.0859375, 0., 0.15625, 0.21875, 0., 0., 0.203125, -0.1328125, 0.1015625, 0.1015625, 0.234375, 0.0859375, 0., -0.0390625, 0., 0., 0.3359375, 0., 0., 0., 0.4765625, -0.3046875, 0., 0.1171875, 0.3515625, -0.21875, 0., 0., -0.28125, 0., 0.515625, 0., 0., 0.3359375, 0., 0., 0.234375, 0., 0., 0., 0.3359375, -0.265625, -0.234375, -0.4296875, 0.4765625, 0., 0., 0.3125, -0.28125, -0.0546875, -0.3828125, 0., 0., -0.4765625, -0.3515625, 0.9453125, 0., 0.265625, -0.5859375, 0., 0., 0.359375, -0.5078125, 0., -0.109375, -0.3359375, 0., 0.578125, -0.9921875, -0.1484375, -0.328125, 0.46875, 0., -0.0625, 0., -0.125, 0., -0.875, -0.4453125, 0.09375, 0., 0.3984375, 0.8671875, -0.453125, 0., 0., -0.03125, 0., -0.2890625, 0., -0.03125, 0., 0., 0., 0., -0.0234375, 0., 0., -0.46875, 0., -0.0625, 0., -0.4140625, 0.34375, -0.4375, 0., 0., -0.5, -0.3515625, 0.0078125, 0., 0., -0.15625, -0.703125, 0.078125, 0., 0., 0., 0., 0., -0.25, 0.0390625, 0., 0.21875, 0.2734375, -0.265625, 0.1015625, 0., 0., 0., -0.8359375, 0., -0.9921875, -0.015625, 0., -0.0234375, 0., 0., 0., 0.3046875, 0., 0., 0., -0.125, -0.140625, -0.2734375, 0.546875, 0.2265625, -0.0625, -0.3984375, 0., 0., 0., -0.2890625, 0., -0.9921875, -0.390625, -0.3359375, 0., 0., 0.046875, 0., -0.2734375, -0.7265625, -0.640625, 0.3359375, 0., 0.6328125, 0., 0.296875, 0., 0.3359375, -0.1953125, -0.1953125, 0.2265625, 0., -0.1171875, 0., 0.375, 0.3515625, 0., 0., 0., 0., 0., 0., 0., -0.34375, 0.0078125, -0.1796875, 0.0703125, 0., -0.03125, 0.0234375, 0., 0.3515625, 0., -0.2734375, 0., 0., -0.234375, -0.421875, 0., 0., -0.1875, -0.390625, 0.21875, 0.375, 0., 0.1484375, 0.3515625, 0.75, 0., -0.1875, 0.6015625, 0.5, 0.25, 0.265625, -0.5546875, 0.8515625, -0.0546875, 0., -0.375, 0.2421875, 0.9921875, -0.2578125, -0.40625, -0.578125, 0.078125, 0.5703125, 0.2109375, 0.6484375, 0.9921875, -0.21875, 0., 0., 0.375, 0.125, -0.3515625, -0.3828125, -0.296875, 0., -0.375, 0.3203125, -0.3046875, 0., 0., 0.171875, 0., -0.25, 0., 0.1640625, -0.5, 0., 0.125, 0., -0.1484375, 0.3203125, 0.2265625, -0.6484375, -0.09375, 0., 0., 0.1640625, 0., 0., -0.109375, 0.265625, 0., 0., 0., 0.0390625, -0.28125, 0., 0.9921875, 0., -0.1875, -0.375, -0.3515625, 0.5703125, 0.625, 0., 0., 0., 0.328125, 0.2578125, -0.671875, 0., 0.328125, 0.3671875, 0., -0.1015625, -0.9921875, 0., 0.40625, -0.3828125, 0.6953125, 0., 0.4296875, -0.125, -0.2265625, 0., 0.2578125, 0., 0., 0., -0.2265625, -0.984375, 0.109375, -0.25, -0.078125, 0., 0., 0.296875, 0., 0.9921875, 0.9921875, 0., -0.3359375, 0., 0.0546875, -0.171875, 0., 0., -0.515625, 0., 0.1640625, 0.2578125, 0., -0.484375, 0., 0., 0.296875, -0.375, 0.1484375, -0.1875, -0.1484375, -0.390625, 0., 0., 0., 0.1171875, 0., 0., 0.3671875, -0.421875, -0.4921875, -0.0546875, 0., 0., 0., 0., -0.0703125, 0., 0., 0., -0.9921875, 0., 0.84375, 0.2578125, 0., 0., 0., -0.0546875, -0.109375, -0.375, 0., -0.4921875, 0., 0.5078125, 0., 0., -0.28125, 0., -0.1171875, 0., 0.1171875, 0., 0.3046875, 0.2890625, -0.8046875, -0.4765625, 0.9375, 0.4609375, -0.703125, -0.4921875, 0.9921875, -0.9921875, 0.9921875, 0.9921875, -0.5703125, -0.9921875, -0.28125, 0.9921875, -0.2109375, 0.1875, -0.1796875, 0.125, -0.9921875, -0.9921875, 0.4453125, 0.1171875, 0.359375, -0.3984375, -0.9921875, 0., 0.0546875, 0., -0.3203125, 0.3359375, -0.5703125, 0., 0., 0., 0., -0.25, 0.9921875, -0.1640625, 0., 0., -0.9921875, 0.078125, 0.390625, 0.1484375, 0.5078125, 0., 0., 0., 0., -0.90625, 0., -0.2109375, -0.0703125, 0., 0., 0.5703125, 0., 0., -0.1875, -0.1328125, 0., 0., -0.2109375, -0.1640625, -0.375, 0., 0., 0., 0., 0., -0.046875, 0., 0.0390625, 0.2734375, 0., 0., 0.9921875, 0., -0.1796875, 0., 0., 0., 0., 0., -0.3046875, 0.1171875, -0.9921875, 0.1328125, 0., -0.3515625, 0.9921875, 0.9921875, -0.234375, 0.1015625, 0., 0.9921875, 0.3203125, 0.4140625, -0.21875, 0., 0.5, 0.3515625, -0.515625, -0.9921875, -0.9921875, -0.6171875, 0.2421875, 0.796875, 0.140625, -0.7734375, 0.2109375, -0.1796875, 0.1875, -0.0703125, 0.5390625, 0.71875, -0.3359375, 0., -0.484375, 0.453125, 0.2265625, 0., 0.4609375, 0., 0., 0.3046875, -0.0859375, 0.1796875, 0.53125, 0.1953125, 0., 0.3125, 0., -0.21875, 0., 0., -0.9921875, 0., -0.1875, -0.1328125, 0.9921875, 0., 0., -0.1328125, 0., -0.8515625, 0.5625, 0.0546875, 0.3671875, 0.375, 0., 0., 0.6171875, 0., 0., 0., 0., -0.9921875, -0.9921875, 0.3125, 0., 0.9921875, -0.6640625, 0., 0., 0., 0., 0.125, -0.9921875, -0.0625, 0.40625, 0., 0.9921875, -0.3828125, 0., 0., 0., 0., -0.484375, 0., 0.3671875, 0.09375, 0.0390625, 0.4921875, 0., -0.03125};
static biases_t biases[OUTPUT_WIDTH] = {-0.1953125, 0.421875, 0.0234375, 0.0703125, 0.4453125, -0.3515625, 0.3515625, 0.328125, 0.25, 0., -0.109375, -0.0703125, 0.0625, -0.078125, 0.0390625, 0.0625, 0.234375, 0.15625, -0.1015625, 0.0390625, 0.0546875, 0.2578125, 0.28125, 0.1953125, 0.2734375, 0.3046875, -0.0625, 0.34375, 0.046875, 0.0078125, 0.1953125, -0.2578125};

void dut(hls::stream<array<input_t, INPUT_WIDTH>> &inputStream,
         hls::stream<array<output_t, OUTPUT_WIDTH>> &outputStream) {
    #pragma HLS dataflow
    #pragma HLS INTERFACE mode=axis port=inputStream
    #pragma HLS INTERFACE mode=axis port=outputStream

    dense<input_t,
          output_t,
          weight_t,
          biases_t,
          accum_t,
          INPUT_WIDTH,
          OUTPUT_WIDTH,
          II>(inputStream,
              outputStream,
              weights,
              biases);

}

```

Writing a unit test is equally simple. I generated the golden reference values using a Python model:

```c++
#include <iostream>
#include "dut.h"

using std::array;

int main()
{
    bool fail = false;

    array<int, INPUT_WIDTH> stimuli = {2,7,9,0,44,0,0,31,14,2,0,0,49,0,1,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    hls::stream<array<input_t, INPUT_WIDTH>> inputStream;

    array<input_t,INPUT_WIDTH> input;
    for(int f = 0; f < INPUT_WIDTH;f++)
        input[f].range(input_t::width-1,0) = stimuli[f];
    inputStream << input;

    hls::stream<array<output_t, OUTPUT_WIDTH>> outputStream;

    dut(inputStream, outputStream);

    array<int, OUTPUT_WIDTH> golden = {59,33,42,32,10,14,46,29,18,58,0,72,61,54,21,34,25,34,0,0,0,0,0,73,68,61,11,0,0,2,57,0};

    array<output_t, OUTPUT_WIDTH> output;
    outputStream >> output;
 
    for(int i = 0; i < OUTPUT_WIDTH; i++) {
        int test_value = output[i].range(output_t::width-1,0);
        if(golden[i] != test_value) {
            std::cout << "Expected output[" << i << "]: golden " << golden[i] << ", got " << test_value << std::endl;
            fail = true;
        }
    }

    if(fail) {
        std::cout << "TEST FAILED" << std::endl;
        return 1;
    } else {
        std::cout << "TEST PASSED" << std::endl;
        return 0;
    }
}

```

Running the C-Simulation, we get the expected result:

```bash
$ ./build/csim.exe
TEST PASSED
INFO [HLS SIM]: The maximum depth reached by any hls::stream() instance in the design is 1
```

So far, so good.

## The Hunt Begins

For further validation, the next step is to run Co-Simulation. Vitis HLS Co-Simulation connects the C-level testbench directly to the generated RTL, allowing us to observe all signals inside the hardware module and verify that the synthesis produced correct logic.

However, when running Co-Simulation, we get the following output:

```bash
$ make cosim

...

INFO: [Common 17-206] Exiting xsim at Sun Dec 14 10:37:39 2025...
INFO: [COSIM 212-316] Starting C post checking ...
Expected output[3]: golden 32, got 25
Expected output[8]: golden 18, got 16
TEST FAILED
```

This is unexpected. We would normally expect a 100% match between transaction-level simulation and cycle-accurate simulation, especially for such a simple piece of code. Yet here we are, with two mismatched outputs.

At this point, the natural instinct is to start debugging our code and look for common mistakes—corner cases where compilers may produce undefined behavior, uninitialized variables, or subtle type mismatches. In this case, however, I can assure you: there was nothing wrong with the code itself.

## Resolving the Issue

As it turns out, the fix is straightforward if you know where to look.

Before explaining the solution, let me provide some technical context. In HLS configuration, AMD offers numerous options for controlling how hardware modules are generated. By default, Vitis HLS infers Digital Signal Processor (DSP) macros on the FPGA for multiplication operations where applicable. These hard DSP macros are particularly valuable when multiplying wide fixed-point numbers with more than 6 or 7 bits per operand, because the alternative of mapping multiplications to distributed logic scales non-linearly with larger bitwidths. Both options exist, and by default, a heuristic algorithm decides how to map each operation onto hardware.

Our use case opens up an even wider range of optimization possibilities. In a Dense Layer, we have one dynamic input for features and one constant input for weights. This asymmetry enables techniques like constant propagation to eliminate unnecessary multiplications with zeros. Vitis HLS can also infer add-and-shift trees for specific constant operands, which proves more efficient than native multiplication via lookup tables.

With such a large design space to cover, it's not surprising that things can go wrong in this part of the HLS toolchain. Suspecting this was the root cause, I experimented with constraining how multiplication operations are mapped onto hardware. We can do this by modifying the configuration file hls.cfg, which orchestrates Vitis HLS:

```ini
[hls]
syn.op=op:mul impl:fabric
```

This directive forces the tool to implement multiplication operations using FPGA fabric with LUTs and registers instead of DSP macros. While this trades off some efficiency since DSPs are optimized for multiply-accumulate operations, it sidesteps the hard DSP macro inference path entirely.

To our relief, the Co-Simulation now passes:

```bash
$ make cosim
TEST PASSED
```

## Explain Me Like I am Five Years Old

Imagine you want to visit your new friend Sophia on the other side of the country. She just moved there, and you're figuring out how to make the trip. You have two options: bus or ferry.

Both are overnight journeys. Both promise to deliver you to the same city. You check the routes, compare the schedules—everything lines up.
You pick the bus. It's faster, even if the route is a bit longer. You settle into your seat, watch the sun set over the hills, and eventually drift off to sleep.

The next morning, you wake up as the bus rolls to a stop. You grab your bag, step outside, stretch your legs... and nothing looks right. The signs are wrong. The streets are unfamiliar.
You're in a completely different city.

How did that happen? You booked the right ticket. You did everything correctly. Turns out the bus driver had a (very) rough night and took a wrong turn somewhere in the dark. And since you were asleep, you couldn't catch the mistake until it was too late.

That's what happened with our neural network.

## Discussion and Conclusion

We encountered a case where AMD Vitis HLS generated functionally incorrect RTL when using DSP macro inference for fixed-point multiplication. The bug manifested as mismatched outputs between C-Simulation and Co-Simulation, with no indication of the root cause in the synthesis reports. Forcing fabric-based multiplication via `syn.op=op:mul impl:fabric` resolved the issue.

This experience highlights that even mature, commercially available HLS tools can contain subtle bugs. Resolving such issues without vendor support is difficult, and nearly impossible without access to source code. In this case, we were fortunate that the bug was reproducible in simulation. Had it only appeared in hardware, diagnosis would have been far more challenging.

Some additional observations:

- Our template closely resembles those used in frameworks like hls4ml and FINN. Similar bugs may affect users of these tools.
- We tested various combinations of data types and numeric values in our dense layer template. We had encountered similar errors previously but it seems to occur only for very specific input configurations. Unfortunately, it is very hard to further debug without original source code.
- AMD responded to our bug report, which exceeded my expectations for vendor engagement on such issues.

This bug was reported to AMD a few weeks ago, and a Change Request has been opened by the AMD support team. You can follow the discussion [here](https://adaptivesupport.amd.com/s/question/0D5Pd000014XT0aKAG/bug-bug-report-for-vitis-hls-20242-on-versal-vck190-when-using-dsps?language=en_US).





