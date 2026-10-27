"""Native representations of the LawSpec conformance domain; no test assertions."""

from dataclasses import make_dataclass
from fractions import Fraction


class Lowering:
    pass


Native = make_dataclass("Native", [], bases=(Lowering,), frozen=True)
Expanded = make_dataclass("Expanded", [], bases=(Lowering,), frozen=True)


class Precision:
    pass


Float32 = make_dataclass("Float32", [], bases=(Precision,), frozen=True)
Float64 = make_dataclass("Float64", [], bases=(Precision,), frozen=True)


class Kernel:
    pass


Mlp = make_dataclass("Mlp", [], bases=(Kernel,), frozen=True)
Convolution = make_dataclass("Convolution", [], bases=(Kernel,), frozen=True)
Normalization = make_dataclass("Normalization", [], bases=(Kernel,), frozen=True)
Softmax = make_dataclass("Softmax", [], bases=(Kernel,), frozen=True)
MaxPool = make_dataclass("MaxPool", [], bases=(Kernel,), frozen=True)
AverageExcludePad = make_dataclass(
    "AverageExcludePad", [], bases=(Kernel,), frozen=True
)
AverageIncludePad = make_dataclass(
    "AverageIncludePad", [], bases=(Kernel,), frozen=True
)


class Cell:
    pass


Rnn = make_dataclass("Rnn", [], bases=(Cell,), frozen=True)
Gru = make_dataclass("Gru", [], bases=(Cell,), frozen=True)
Lstm = make_dataclass("Lstm", [], bases=(Cell,), frozen=True)


class Mask:
    pass


BooleanMask = make_dataclass("BooleanMask", [], bases=(Mask,), frozen=True)
AdditiveMask = make_dataclass("AdditiveMask", [], bases=(Mask,), frozen=True)
CausalMask = make_dataclass("CausalMask", [], bases=(Mask,), frozen=True)
FullyMasked = make_dataclass("FullyMasked", [], bases=(Mask,), frozen=True)


class Expansion:
    pass


LinearExpansion = make_dataclass("LinearExpansion", [], bases=(Expansion,), frozen=True)
SoftmaxExpansion = make_dataclass(
    "SoftmaxExpansion", [], bases=(Expansion,), frozen=True
)
EmptySoftmaxExpansion = make_dataclass(
    "EmptySoftmaxExpansion", [], bases=(Expansion,), frozen=True
)
ConvExpansion = make_dataclass("ConvExpansion", [], bases=(Expansion,), frozen=True)
MaxPoolExpansion = make_dataclass(
    "MaxPoolExpansion", [], bases=(Expansion,), frozen=True
)
AvgPoolExpansion = make_dataclass(
    "AvgPoolExpansion", [], bases=(Expansion,), frozen=True
)
LayerNormExpansion = make_dataclass(
    "LayerNormExpansion", [], bases=(Expansion,), frozen=True
)
BatchNormExpansion = make_dataclass(
    "BatchNormExpansion", [], bases=(Expansion,), frozen=True
)
AttentionExpansion = make_dataclass(
    "AttentionExpansion", [], bases=(Expansion,), frozen=True
)
DropoutExpansion = make_dataclass(
    "DropoutExpansion", [], bases=(Expansion,), frozen=True
)


class ImportedLibrary:
    pass


ImportedBatchNorm = make_dataclass(
    "ImportedBatchNorm", [], bases=(ImportedLibrary,), frozen=True
)
ImportedEmbedding = make_dataclass(
    "ImportedEmbedding", [], bases=(ImportedLibrary,), frozen=True
)
ImportedPool = make_dataclass("ImportedPool", [], bases=(ImportedLibrary,), frozen=True)


class Rejection:
    pass


DynamicCapture = make_dataclass("DynamicCapture", [], bases=(Rejection,), frozen=True)
UnknownCapture = make_dataclass("UnknownCapture", [], bases=(Rejection,), frozen=True)
WrongRank = make_dataclass("WrongRank", [], bases=(Rejection,), frozen=True)
WrongDtype = make_dataclass("WrongDtype", [], bases=(Rejection,), frozen=True)


class Mutation:
    pass


ReverseMask = make_dataclass("ReverseMask", [], bases=(Mutation,), frozen=True)
DuplicateParameter = make_dataclass(
    "DuplicateParameter", [], bases=(Mutation,), frozen=True
)
BiasedVariance = make_dataclass("BiasedVariance", [], bases=(Mutation,), frozen=True)
ExecuteBothBranches = make_dataclass(
    "ExecuteBothBranches", [], bases=(Mutation,), frozen=True
)


class ShapeOperation:
    pass


BroadcastShape = make_dataclass(
    "BroadcastShape", [], bases=(ShapeOperation,), frozen=True
)
MatmulShape = make_dataclass("MatmulShape", [], bases=(ShapeOperation,), frozen=True)
ReshapeShape = make_dataclass("ReshapeShape", [], bases=(ShapeOperation,), frozen=True)
SumShape = make_dataclass("SumShape", [], bases=(ShapeOperation,), frozen=True)
MeanShape = make_dataclass("MeanShape", [], bases=(ShapeOperation,), frozen=True)
MaximumShape = make_dataclass("MaximumShape", [], bases=(ShapeOperation,), frozen=True)
MinimumShape = make_dataclass("MinimumShape", [], bases=(ShapeOperation,), frozen=True)
ExpShape = make_dataclass("ExpShape", [], bases=(ShapeOperation,), frozen=True)
SqrtShape = make_dataclass("SqrtShape", [], bases=(ShapeOperation,), frozen=True)
DivideShape = make_dataclass("DivideShape", [], bases=(ShapeOperation,), frozen=True)


class ShapeDtype:
    pass


ShapeFloat32 = make_dataclass("ShapeFloat32", [], bases=(ShapeDtype,), frozen=True)
ShapeFloat64 = make_dataclass("ShapeFloat64", [], bases=(ShapeDtype,), frozen=True)
ShapeInt64 = make_dataclass("ShapeInt64", [], bases=(ShapeDtype,), frozen=True)
ShapeBool = make_dataclass("ShapeBool", [], bases=(ShapeDtype,), frozen=True)


class AxisChoice:
    pass


AllAxes = make_dataclass("AllAxes", [], bases=(AxisChoice,), frozen=True)
FirstAxis = make_dataclass("FirstAxis", [], bases=(AxisChoice,), frozen=True)
LastAxis = make_dataclass("LastAxis", [], bases=(AxisChoice,), frozen=True)
EmptyAxes = make_dataclass("EmptyAxes", [], bases=(AxisChoice,), frozen=True)


class Sample:
    pass


Finite = make_dataclass("Finite", [("value", Fraction)], bases=(Sample,), frozen=True)
NotANumber = make_dataclass("NotANumber", [], bases=(Sample,), frozen=True)
PositiveInfinity = make_dataclass("PositiveInfinity", [], bases=(Sample,), frozen=True)
NegativeInfinity = make_dataclass("NegativeInfinity", [], bases=(Sample,), frozen=True)
KernelCase = make_dataclass(
    "KernelCase",
    [("kernel", object), ("precision", object), ("lowering", object)],
    frozen=True,
)
SharedCase = make_dataclass(
    "SharedCase", [("precision", object), ("lowering", object)], frozen=True
)
CellCase = make_dataclass(
    "CellCase", [("cell", object), ("lowering", object)], frozen=True
)
AttentionCase = make_dataclass(
    "AttentionCase",
    [("mask", object), ("precision", object), ("lowering", object)],
    frozen=True,
)
ExpansionCase = make_dataclass(
    "ExpansionCase", [("expansion", object), ("lowering", object)], frozen=True
)
ImportCase = make_dataclass(
    "ImportCase", [("library", object), ("lowering", object)], frozen=True
)
ShapeCase = make_dataclass(
    "ShapeCase",
    [
        ("operation", object),
        ("left", object),
        ("right", object),
        ("target", object),
        ("dtype", object),
        ("axis", object),
    ],
    frozen=True,
)
TensorObservation = make_dataclass(
    "TensorObservation",
    [("path", object), ("shape", object), ("dtype", object), ("samples", object)],
    frozen=True,
)
Observation = make_dataclass(
    "Observation", [("tensors", object), ("facts", object)], frozen=True
)


class ConformanceCase:
    pass


ModuleExecution = make_dataclass(
    "ModuleExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
SharedExecution = make_dataclass(
    "SharedExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
BatchState = make_dataclass(
    "BatchState", [("value", object)], bases=(ConformanceCase,), frozen=True
)
CellExecution = make_dataclass(
    "CellExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
EmbeddingExecution = make_dataclass(
    "EmbeddingExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
AttentionExecution = make_dataclass(
    "AttentionExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
DropoutExecution = make_dataclass(
    "DropoutExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
SoftmaxBoundaries = make_dataclass(
    "SoftmaxBoundaries", [("value", object)], bases=(ConformanceCase,), frozen=True
)
ControlExecution = make_dataclass(
    "ControlExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
CollapseExecution = make_dataclass(
    "CollapseExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
SavedExport = make_dataclass(
    "SavedExport", [("value", object)], bases=(ConformanceCase,), frozen=True
)
CapturedExport = make_dataclass(
    "CapturedExport", [("value", object)], bases=(ConformanceCase,), frozen=True
)
RejectedExecution = make_dataclass(
    "RejectedExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
SequenceExecution = make_dataclass(
    "SequenceExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
ExpandedExecution = make_dataclass(
    "ExpandedExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
SavedConditional = make_dataclass(
    "SavedConditional", [("value", object)], bases=(ConformanceCase,), frozen=True
)
SavedRecurrence = make_dataclass(
    "SavedRecurrence", [("value", object)], bases=(ConformanceCase,), frozen=True
)
CapturedConditional = make_dataclass(
    "CapturedConditional", [("value", object)], bases=(ConformanceCase,), frozen=True
)
ImportedExecution = make_dataclass(
    "ImportedExecution", [("value", object)], bases=(ConformanceCase,), frozen=True
)
NamedAxes = make_dataclass("NamedAxes", [], bases=(ConformanceCase,), frozen=True)
FrozenState = make_dataclass("FrozenState", [], bases=(ConformanceCase,), frozen=True)
