//===-- AMDGPU.td - AMDGPU dialect definitions *- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPU
#define AMDGPU

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def AMDGPU_Dialect : Dialect {
  let name = "amdgpu";
  let cppNamespace = "::mlir::amdgpu";
  let description = [{
    The `AMDGPU` dialect provides wrappers around AMD-specific functionality
    and LLVM intrinsics. These wrappers should be used in conjunction with
    more generic dialects, such as `gpu` and `vector`, when generating LLVM IR
    that will eventually be executed on AMD hardware.
  }];


  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// AMDGPU Op definitions
//===----------------------------------------------------------------------===//

class AMDGPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<AMDGPU_Dialect, mnemonic, traits> {}

/// Raw buffer load
def AMDGPU_RawBufferLoadOp :
    AMDGPU_Op<"raw_buffer_load", [AllElementTypesMatch<["value", "memref"]>,
      AttrSizedOperandSegments]>,
    Arguments<(ins Arg<AnyMemRef, "buffer to load from", [MemRead]>:$memref,
                   Variadic<I32>:$indices,
                   DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
                   OptionalAttr<I32Attr>:$indexOffset,
                   Optional<I32>:$sgprOffset)>,
    Results<(outs AnyTypeOf<[BF16, F16, F32, I32, I8,
                              VectorOfLengthAndType<[2, 4], [F32, I32]>,
                              VectorOfLengthAndType<[2, 4, 8], [F16, BF16]>,
                              VectorOfLengthAndType<[2, 4, 8, 16], [I8]>]>:$value)> {

  let summary = "Raw Buffer load, exposing GCN features";
  let description = [{
    The `amdgpu.raw_buffer_load` op is a wrapper around the buffer load intrinsics
    available on AMD GPUs, including extensions in newer GPUs.

    The index into the buffer is computed as for `memref.load` with the additon
    of `indexOffset` and `sgprOffset` (which **may or may not** be considered
    in bounds checks and includes any offset present on the memref type if it's
    non-zero).

    All indices and offsets are in units of the memref's data type and are
    converted to bytes during lowering.

    When a load is out of bounds, the instruction returns zero.
    Partially-out of bounds have chipset-dependent behavior: whether reading
    2 elements starting at index 7 of a `memref<8xf32>` returns the last element
    in the first vector component depends on the architecture.

    The memref struct is converted into a buffer resource (a V#) and the arguments
    are translated to intrinsic arguments as follows:
    - The base address of the buffer is the base address of the memref
    - The stride is 0 to enable raw mode
    - The number of records is the size of the memref, in bytes
      In the case of dynamically-shaped memrefs, this is computed at runtime
      as max_d (size(d) * stride(d)) * sizeof(elementType(memref))
    - The offset enable bit is 1, the index enable bit is 0.
    - The thread ID addition bit is off
    - If `boundsCheck` is false and the target chipset is RDNA, OOB_SELECT is set
      to 2 to disable bounds checks, otherwise it is 3
    - The cache coherency bits are off
  }];
  let assemblyFormat = [{
    attr-dict $memref `[` $indices `]`
      (`sgprOffset` $sgprOffset^)? `:`
      type($memref) `,` type($indices) `->` type($value)
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

/// Raw buffer store
def AMDGPU_RawBufferStoreOp :
    AMDGPU_Op<"raw_buffer_store", [AllElementTypesMatch<["value", "memref"]>,
      AttrSizedOperandSegments]>,
    Arguments<(ins AnyTypeOf<[BF16, F16, F32, I32, I8,
                              VectorOfLengthAndType<[2, 4], [F32, I32]>,
                              VectorOfLengthAndType<[2, 4, 8], [F16, BF16]>,
                              VectorOfLengthAndType<[2, 4, 8, 16], [I8]>]>:$value,
                   Arg<AnyMemRef, "buffer to store to", [MemWrite]>:$memref,
                   Variadic<I32>:$indices,
                   DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
                   OptionalAttr<I32Attr>:$indexOffset,
                   Optional<I32>:$sgprOffset)> {

  let summary = "Raw Buffer Store, exposing GCN features";
  let description = [{
    The `amdgpu.raw_buffer_store` op is a wrapper around the buffer store
    intrinsics available on AMD GPUs, including extensions in newer GPUs.

    The store index is computed as in `memref.store` with the addition of
    `indexOffset` (which is included for uniformity with atomics and may be useful
    when writing vectorized code) and `sgprOffset` (which is added after bounds
    checks and implicitly includes the offset of the memref type if non-zero).
    All index components are in terms of the elements of the memref, not bytes,
    and are scaled up appropriately.

    Out of bounds stores are ignored in hardware.
    Wthether a vector write that includes some in-bounds and soeme out-of-bounds
    components is partically completed is chipset-dependent.

    See `amdgpu.raw_buffer_load` for a description of how the underlying
    instruction is constructed.
  }];
  let assemblyFormat = [{
    attr-dict $value `->` $memref `[` $indices `]`
      (`sgprOffset` $sgprOffset^)? `:`
      type($value) `->` type($memref) `,` type($indices)
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

// Raw buffer atomic floating point add
def AMDGPU_RawBufferAtomicFaddOp :
    AMDGPU_Op<"raw_buffer_atomic_fadd", [AllElementTypesMatch<["value", "memref"]>,
      AttrSizedOperandSegments]>,
    Arguments<(ins F32:$value,
                   Arg<AnyMemRef, "buffer to operate on", [MemRead, MemWrite]>:$memref,
                   Variadic<I32>:$indices,
                   DefaultValuedAttr<BoolAttr, "true">:$boundsCheck,
                   OptionalAttr<I32Attr>:$indexOffset,
                   Optional<I32>:$sgprOffset)> {

  let summary = "Raw Buffer Floating-point Atomic Add (MI-* only)";
  let description = [{
    The `amdgpu.raw_buffer_atomic_fadd` op is a wrapper around the
    buffer-based atomic floating point addition available on the MI-* series
    of AMD GPUs.

    The index into the buffer is computed as for `memref.store` with the addition
    of `indexOffset` (which is used to aid in emitting vectorized code) and,
    if present `sgprOffset` (which is added after bounds checks and includes
    any non-zero offset on the memref type).

    All indexing components are given in terms of the memref's element size, not
    the byte lengths required by the intrinsic.

    Out of bounds atomic operations are ignored in hardware.

    See `amdgpu.raw_buffer_load` for a description of how the underlying
    instruction is constructed.
  }];
  let assemblyFormat = [{
    attr-dict $value `->` $memref `[` $indices `]`
      (`sgprOffset` $sgprOffset^)? `:`
      type($value) `->` type($memref) `,` type($indices)
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def AMDGPU_LDSBarrierOp : AMDGPU_Op<"lds_barrier"> {
  let summary = "Barrier that includes a wait for LDS memory operations.";
  let description = [{
    `amdgpu.lds_barrier` is both a barrier (all workitems in a workgroup must reach
    the barrier before any of them may proceed past it) and a wait for all
    operations that affect the Local Data Store (LDS) issued from that wrokgroup
    to complete before the workgroup may continue. Since the LDS is per-workgroup
    memory, this barrier may be used, for example, to ensure all workitems have
    written data to LDS before any workitem attempts to read from it.

    Note that `lds_barrier` does **not** force reads to or from global memory
    to complete before execution continues. Therefore, it should be used when
    operations on global memory can be issued far in advance of when their results
    are used (for example, by writing them to LDS).
  }];
  let assemblyFormat = "attr-dict";
}

def AMDGPU_MFMAPermB : I32EnumAttr<"MFMAPermB",
    "The possible permutations of the lanes storing B available in an MFMA",
    [
      I32EnumAttrCase<"none",            0>,
      I32EnumAttrCase<"bcast_first_32",  1>,
      I32EnumAttrCase<"bcast_second_32", 2>,
      I32EnumAttrCase<"rotate_16_right", 3>,
      I32EnumAttrCase<"bcast_first_16",  4>,
      I32EnumAttrCase<"bcast_second_16", 5>,
      I32EnumAttrCase<"bcast_third_16",  6>,
      I32EnumAttrCase<"bcast_fourth_16", 7>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::amdgpu";
}

def AMDGPU_MFMAPermBAttr : EnumAttr<AMDGPU_Dialect, AMDGPU_MFMAPermB,
  "mfma_perm_b">;

// mfma
def MFMAInTypes : AnyTypeOf<[F32, F64, I32, I64,
                             VectorOfLengthAndType<[2], [F32]>,
                             VectorOfLengthAndType<[4], [F16]>,
                             VectorOfLengthAndType<[2, 4], [BF16]>,
                             VectorOfLengthAndType<[4, 8], [I8]>]>;
def MFMAOutTypes : AnyTypeOf<[F64,
                              VectorOfLengthAndType<[4, 16, 32], [F32]>,
                              VectorOfLengthAndType<[4, 16, 32], [I32]>,
                              VectorOfLengthAndType<[4], [F64]>]>;

def AMDGPU_MFMAOp :
    AMDGPU_Op<"mfma", [AllTypesMatch<["sourceA", "sourceB"]>,
                        AllTypesMatch<["destC", "destD"]>,
                        Pure]>,
    Arguments<(ins
                   I32Attr:$m,
                   I32Attr:$n,
                   I32Attr:$k,
                   I32Attr:$blocks,
                   MFMAInTypes:$sourceA,
                   MFMAInTypes:$sourceB,
                   MFMAOutTypes:$destC,
                   DefaultValuedAttr<I32Attr, "0">:$cbsz,
                   DefaultValuedAttr<I32Attr, "0">:$abid,
                   DefaultValuedAttr<AMDGPU_MFMAPermBAttr,
                    "::mlir::amdgpu::MFMAPermB::none">:$blgp,
                   UnitAttr:$reducePrecision,
                   UnitAttr:$negateA,
                   UnitAttr:$negateB,
                   UnitAttr:$negateC)>,
    Results<(outs MFMAOutTypes: $destD)> {
  let summary = "MLIR wrapper for CDNA mfma instructions";
  let description = [{
    The `amdgpu.mfma` op is an MLIR wrapper around intrinsics
    for various `mfma` instructions in the CDNA architecture, which perform
    multiple outer products in order to allow fast matrix multiplication.

    The wrapper will select an appropriate `mfma` instruction, if one is available,
    based on the provided `m`, `k`, `n`, and `nBlks` attributes, along with the
    types of the source and destination arguments.

    For information on the layouts of the input and output matrces (which are stored
    in `sourceA`, `sourceB`, `destC`, and `destD`), see the CDNA ISA documentation.

    The `cbsz`, `abid`, and `blgp` parameters control how the lanes of the wave
    are permuted when matrix data is being loaded: `blgp` can be any number of
    fixed permutations, `cbsz` specifies the log_2 of the number of chunks the lanes
    holding sourceA are split into, and `abid` selects one of those chunks.

    Note, this wrapper allows specifying `vector<4Kxi8>` arguments to MFMA
    intrinsics that take an integer type of width `4K`. For example,
    one can provide a vector<4xi8> as an argument to an MFMA instruction that
    logically takes 4 i8s but whose intrinsics are specified to take an i32.
    In these cases, the bytes in the vector will be concatenated in little-endian
    order (that is, v[0] will go to arg[7:0], v[1] to arg[15:8] and so on).

    The negateA, negateB, and negateC flags are only supported for double-precision
    operations on gfx940+.
  }];
  let assemblyFormat = [{
    $sourceA `*` $sourceB `+` $destC
    attr-dict
    `blgp` `=` $blgp
    `:` type($sourceA) `,` type($destC)
  }];
  let hasVerifier = 1;
}

#endif // AMDGPU
