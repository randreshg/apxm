# Glossary

## AIS
Agent Instruction Set. The canonical set of operations that define agent programs.

## AAM
Agent Abstract Machine. The runtime memory model (STM/LTM/Episodic) and execution context.

## Artifact
The compiled binary format emitted by the compiler and consumed by the runtime.

## ExecutionDag
A directed acyclic graph of AIS operations, inputs, and dependencies.

## Pass Pipeline
The sequence of compiler passes that normalize, schedule, and lower AIS MLIR.

## TableGen
MLIR TableGen files generated from `apxm-ais` operation metadata.

## Operation Metadata
Static description of each AIS op (name, fields, required inputs, traits).

## Runtime Scheduler
The dataflow scheduler that executes ready nodes in parallel.
