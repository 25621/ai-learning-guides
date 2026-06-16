# FPGA Inference

## Key Insight

A [Field-Programmable Gate Array (FPGA)](/shared/glossary/#fpga) offers a middle ground between a general-purpose [CPU](/shared/glossary/#cpu)/[GPU](/shared/glossary/#gpu) and a fixed-function [ASIC](/shared/glossary/#asic): its logic gates can be rewired after manufacturing to implement a custom hardware accelerator without the multi-million-dollar cost of fabricating a new chip. Designing a small [convolutional neural network](/shared/glossary/#cnn) inference pipeline on an FPGA demonstrates how hardwiring the data movement and arithmetic operations directly into the chip's logic fabric eliminates the instruction-fetch and scheduling overhead that slows down software-based execution. This project exposes developers to hardware description languages (like Verilog) or high-level synthesis (HLS) tools, and the process of compiling a design into a bitstream — the binary configuration file that physically rewires the FPGA's logic cells.
