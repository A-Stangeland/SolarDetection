%
È¬
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	

ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02unknown8ø

conv2d_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_57/kernel
}
$conv2d_57/kernel/Read/ReadVariableOpReadVariableOpconv2d_57/kernel*&
_output_shapes
: *
dtype0
t
conv2d_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_57/bias
m
"conv2d_57/bias/Read/ReadVariableOpReadVariableOpconv2d_57/bias*
_output_shapes
: *
dtype0

conv2d_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_58/kernel
}
$conv2d_58/kernel/Read/ReadVariableOpReadVariableOpconv2d_58/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_58/bias
m
"conv2d_58/bias/Read/ReadVariableOpReadVariableOpconv2d_58/bias*
_output_shapes
: *
dtype0

conv2d_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_59/kernel
}
$conv2d_59/kernel/Read/ReadVariableOpReadVariableOpconv2d_59/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_59/bias
m
"conv2d_59/bias/Read/ReadVariableOpReadVariableOpconv2d_59/bias*
_output_shapes
:@*
dtype0

conv2d_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_60/kernel
}
$conv2d_60/kernel/Read/ReadVariableOpReadVariableOpconv2d_60/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_60/bias
m
"conv2d_60/bias/Read/ReadVariableOpReadVariableOpconv2d_60/bias*
_output_shapes
:@*
dtype0

conv2d_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_61/kernel
~
$conv2d_61/kernel/Read/ReadVariableOpReadVariableOpconv2d_61/kernel*'
_output_shapes
:@*
dtype0
u
conv2d_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_61/bias
n
"conv2d_61/bias/Read/ReadVariableOpReadVariableOpconv2d_61/bias*
_output_shapes	
:*
dtype0

conv2d_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_62/kernel

$conv2d_62/kernel/Read/ReadVariableOpReadVariableOpconv2d_62/kernel*(
_output_shapes
:*
dtype0
u
conv2d_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_62/bias
n
"conv2d_62/bias/Read/ReadVariableOpReadVariableOpconv2d_62/bias*
_output_shapes	
:*
dtype0

conv2d_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_63/kernel

$conv2d_63/kernel/Read/ReadVariableOpReadVariableOpconv2d_63/kernel*(
_output_shapes
:*
dtype0
u
conv2d_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_63/bias
n
"conv2d_63/bias/Read/ReadVariableOpReadVariableOpconv2d_63/bias*
_output_shapes	
:*
dtype0

conv2d_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_64/kernel

$conv2d_64/kernel/Read/ReadVariableOpReadVariableOpconv2d_64/kernel*(
_output_shapes
:*
dtype0
u
conv2d_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_64/bias
n
"conv2d_64/bias/Read/ReadVariableOpReadVariableOpconv2d_64/bias*
_output_shapes	
:*
dtype0

conv2d_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_65/kernel

$conv2d_65/kernel/Read/ReadVariableOpReadVariableOpconv2d_65/kernel*(
_output_shapes
:*
dtype0
u
conv2d_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_65/bias
n
"conv2d_65/bias/Read/ReadVariableOpReadVariableOpconv2d_65/bias*
_output_shapes	
:*
dtype0

conv2d_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_66/kernel

$conv2d_66/kernel/Read/ReadVariableOpReadVariableOpconv2d_66/kernel*(
_output_shapes
:*
dtype0
u
conv2d_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_66/bias
n
"conv2d_66/bias/Read/ReadVariableOpReadVariableOpconv2d_66/bias*
_output_shapes	
:*
dtype0

conv2d_67/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_67/kernel

$conv2d_67/kernel/Read/ReadVariableOpReadVariableOpconv2d_67/kernel*(
_output_shapes
:*
dtype0
u
conv2d_67/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_67/bias
n
"conv2d_67/bias/Read/ReadVariableOpReadVariableOpconv2d_67/bias*
_output_shapes	
:*
dtype0

conv2d_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_68/kernel

$conv2d_68/kernel/Read/ReadVariableOpReadVariableOpconv2d_68/kernel*(
_output_shapes
:*
dtype0
u
conv2d_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_68/bias
n
"conv2d_68/bias/Read/ReadVariableOpReadVariableOpconv2d_68/bias*
_output_shapes	
:*
dtype0

conv2d_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_69/kernel

$conv2d_69/kernel/Read/ReadVariableOpReadVariableOpconv2d_69/kernel*(
_output_shapes
:*
dtype0
u
conv2d_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_69/bias
n
"conv2d_69/bias/Read/ReadVariableOpReadVariableOpconv2d_69/bias*
_output_shapes	
:*
dtype0

conv2d_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_70/kernel

$conv2d_70/kernel/Read/ReadVariableOpReadVariableOpconv2d_70/kernel*(
_output_shapes
:*
dtype0
u
conv2d_70/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_70/bias
n
"conv2d_70/bias/Read/ReadVariableOpReadVariableOpconv2d_70/bias*
_output_shapes	
:*
dtype0

conv2d_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:À@*!
shared_nameconv2d_71/kernel
~
$conv2d_71/kernel/Read/ReadVariableOpReadVariableOpconv2d_71/kernel*'
_output_shapes
:À@*
dtype0
t
conv2d_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_71/bias
m
"conv2d_71/bias/Read/ReadVariableOpReadVariableOpconv2d_71/bias*
_output_shapes
:@*
dtype0

conv2d_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_72/kernel
}
$conv2d_72/kernel/Read/ReadVariableOpReadVariableOpconv2d_72/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_72/bias
m
"conv2d_72/bias/Read/ReadVariableOpReadVariableOpconv2d_72/bias*
_output_shapes
:@*
dtype0

conv2d_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:` *!
shared_nameconv2d_73/kernel
}
$conv2d_73/kernel/Read/ReadVariableOpReadVariableOpconv2d_73/kernel*&
_output_shapes
:` *
dtype0
t
conv2d_73/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_73/bias
m
"conv2d_73/bias/Read/ReadVariableOpReadVariableOpconv2d_73/bias*
_output_shapes
: *
dtype0

conv2d_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_74/kernel
}
$conv2d_74/kernel/Read/ReadVariableOpReadVariableOpconv2d_74/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_74/bias
m
"conv2d_74/bias/Read/ReadVariableOpReadVariableOpconv2d_74/bias*
_output_shapes
: *
dtype0

conv2d_75/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_75/kernel
}
$conv2d_75/kernel/Read/ReadVariableOpReadVariableOpconv2d_75/kernel*&
_output_shapes
: *
dtype0
t
conv2d_75/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_75/bias
m
"conv2d_75/bias/Read/ReadVariableOpReadVariableOpconv2d_75/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:È*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:È*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:È*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:È*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
z
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_positives_1
s
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes
:*
dtype0
x
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_2
q
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes
:*
dtype0
z
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_negatives_1
s
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
:*
dtype0

Adam/conv2d_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_57/kernel/m

+Adam/conv2d_57/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_57/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_57/bias/m
{
)Adam/conv2d_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_57/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_58/kernel/m

+Adam/conv2d_58/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_58/kernel/m*&
_output_shapes
:  *
dtype0

Adam/conv2d_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_58/bias/m
{
)Adam/conv2d_58/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_58/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_59/kernel/m

+Adam/conv2d_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_59/bias/m
{
)Adam/conv2d_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_60/kernel/m

+Adam/conv2d_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_60/bias/m
{
)Adam/conv2d_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_61/kernel/m

+Adam/conv2d_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_61/bias/m
|
)Adam/conv2d_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_62/kernel/m

+Adam/conv2d_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_62/bias/m
|
)Adam/conv2d_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_63/kernel/m

+Adam/conv2d_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_63/bias/m
|
)Adam/conv2d_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_64/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_64/kernel/m

+Adam/conv2d_64/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_64/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_64/bias/m
|
)Adam/conv2d_64/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_65/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_65/kernel/m

+Adam/conv2d_65/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_65/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_65/bias/m
|
)Adam/conv2d_65/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_66/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_66/kernel/m

+Adam/conv2d_66/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_66/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_66/bias/m
|
)Adam/conv2d_66/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_67/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_67/kernel/m

+Adam/conv2d_67/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_67/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_67/bias/m
|
)Adam/conv2d_67/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_68/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_68/kernel/m

+Adam/conv2d_68/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_68/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_68/bias/m
|
)Adam/conv2d_68/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_69/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_69/kernel/m

+Adam/conv2d_69/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_69/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_69/bias/m
|
)Adam/conv2d_69/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_70/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_70/kernel/m

+Adam/conv2d_70/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_70/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_70/bias/m
|
)Adam/conv2d_70/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_71/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:À@*(
shared_nameAdam/conv2d_71/kernel/m

+Adam/conv2d_71/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_71/kernel/m*'
_output_shapes
:À@*
dtype0

Adam/conv2d_71/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_71/bias/m
{
)Adam/conv2d_71/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_71/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_72/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_72/kernel/m

+Adam/conv2d_72/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_72/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_72/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_72/bias/m
{
)Adam/conv2d_72/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_72/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_73/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:` *(
shared_nameAdam/conv2d_73/kernel/m

+Adam/conv2d_73/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/kernel/m*&
_output_shapes
:` *
dtype0

Adam/conv2d_73/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_73/bias/m
{
)Adam/conv2d_73/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_74/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_74/kernel/m

+Adam/conv2d_74/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_74/kernel/m*&
_output_shapes
:  *
dtype0

Adam/conv2d_74/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_74/bias/m
{
)Adam/conv2d_74/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_74/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_75/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_75/kernel/m

+Adam/conv2d_75/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_75/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_75/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_75/bias/m
{
)Adam/conv2d_75/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_75/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_57/kernel/v

+Adam/conv2d_57/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_57/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_57/bias/v
{
)Adam/conv2d_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_57/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_58/kernel/v

+Adam/conv2d_58/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_58/kernel/v*&
_output_shapes
:  *
dtype0

Adam/conv2d_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_58/bias/v
{
)Adam/conv2d_58/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_58/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_59/kernel/v

+Adam/conv2d_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_59/bias/v
{
)Adam/conv2d_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_60/kernel/v

+Adam/conv2d_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_60/bias/v
{
)Adam/conv2d_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_61/kernel/v

+Adam/conv2d_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_61/bias/v
|
)Adam/conv2d_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_62/kernel/v

+Adam/conv2d_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_62/bias/v
|
)Adam/conv2d_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_63/kernel/v

+Adam/conv2d_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_63/bias/v
|
)Adam/conv2d_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_64/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_64/kernel/v

+Adam/conv2d_64/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_64/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_64/bias/v
|
)Adam/conv2d_64/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_65/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_65/kernel/v

+Adam/conv2d_65/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_65/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_65/bias/v
|
)Adam/conv2d_65/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_66/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_66/kernel/v

+Adam/conv2d_66/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_66/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_66/bias/v
|
)Adam/conv2d_66/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_67/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_67/kernel/v

+Adam/conv2d_67/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_67/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_67/bias/v
|
)Adam/conv2d_67/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_68/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_68/kernel/v

+Adam/conv2d_68/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_68/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_68/bias/v
|
)Adam/conv2d_68/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_69/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_69/kernel/v

+Adam/conv2d_69/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_69/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_69/bias/v
|
)Adam/conv2d_69/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_70/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_70/kernel/v

+Adam/conv2d_70/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_70/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_70/bias/v
|
)Adam/conv2d_70/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_71/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:À@*(
shared_nameAdam/conv2d_71/kernel/v

+Adam/conv2d_71/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_71/kernel/v*'
_output_shapes
:À@*
dtype0

Adam/conv2d_71/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_71/bias/v
{
)Adam/conv2d_71/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_71/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_72/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_72/kernel/v

+Adam/conv2d_72/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_72/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_72/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_72/bias/v
{
)Adam/conv2d_72/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_72/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_73/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:` *(
shared_nameAdam/conv2d_73/kernel/v

+Adam/conv2d_73/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/kernel/v*&
_output_shapes
:` *
dtype0

Adam/conv2d_73/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_73/bias/v
{
)Adam/conv2d_73/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_74/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_74/kernel/v

+Adam/conv2d_74/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_74/kernel/v*&
_output_shapes
:  *
dtype0

Adam/conv2d_74/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_74/bias/v
{
)Adam/conv2d_74/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_74/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_75/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_75/kernel/v

+Adam/conv2d_75/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_75/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_75/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_75/bias/v
{
)Adam/conv2d_75/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_75/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Èß
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ß
value÷ÞBóÞ BëÞ

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer-20
layer_with_weights-12
layer-21
layer_with_weights-13
layer-22
layer-23
layer-24
layer_with_weights-14
layer-25
layer_with_weights-15
layer-26
layer-27
layer-28
layer_with_weights-16
layer-29
layer_with_weights-17
layer-30
 layer_with_weights-18
 layer-31
!	optimizer
"regularization_losses
#	variables
$trainable_variables
%	keras_api
&
signatures
 
h

'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
R
3regularization_losses
4	variables
5trainable_variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
R
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
h

Gkernel
Hbias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
h

Mkernel
Nbias
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
R
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
h

Wkernel
Xbias
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
h

]kernel
^bias
_regularization_losses
`	variables
atrainable_variables
b	keras_api
R
cregularization_losses
d	variables
etrainable_variables
f	keras_api
h

gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
h

mkernel
nbias
oregularization_losses
p	variables
qtrainable_variables
r	keras_api
R
sregularization_losses
t	variables
utrainable_variables
v	keras_api
R
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
i

{kernel
|bias
}regularization_losses
~	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
 	variables
¡trainable_variables
¢	keras_api
n
£kernel
	¤bias
¥regularization_losses
¦	variables
§trainable_variables
¨	keras_api
n
©kernel
	ªbias
«regularization_losses
¬	variables
­trainable_variables
®	keras_api
V
¯regularization_losses
°	variables
±trainable_variables
²	keras_api
V
³regularization_losses
´	variables
µtrainable_variables
¶	keras_api
n
·kernel
	¸bias
¹regularization_losses
º	variables
»trainable_variables
¼	keras_api
n
½kernel
	¾bias
¿regularization_losses
À	variables
Átrainable_variables
Â	keras_api
n
Ãkernel
	Äbias
Åregularization_losses
Æ	variables
Çtrainable_variables
È	keras_api
Ý
	Éiter
Êbeta_1
Ëbeta_2

Ìdecay
Ílearning_rate'm(m-m.m7m8m=m>mGmHmMmNmWmXm]m ^m¡gm¢hm£mm¤nm¥{m¦|m§	m¨	m©	mª	m«	m¬	m­	£m®	¤m¯	©m°	ªm±	·m²	¸m³	½m´	¾mµ	Ãm¶	Äm·'v¸(v¹-vº.v»7v¼8v½=v¾>v¿GvÀHvÁMvÂNvÃWvÄXvÅ]vÆ^vÇgvÈhvÉmvÊnvË{vÌ|vÍ	vÎ	vÏ	vÐ	vÑ	vÒ	vÓ	£vÔ	¤vÕ	©vÖ	ªv×	·vØ	¸vÙ	½vÚ	¾vÛ	ÃvÜ	ÄvÝ
 
¶
'0
(1
-2
.3
74
85
=6
>7
G8
H9
M10
N11
W12
X13
]14
^15
g16
h17
m18
n19
{20
|21
22
23
24
25
26
27
£28
¤29
©30
ª31
·32
¸33
½34
¾35
Ã36
Ä37
¶
'0
(1
-2
.3
74
85
=6
>7
G8
H9
M10
N11
W12
X13
]14
^15
g16
h17
m18
n19
{20
|21
22
23
24
25
26
27
£28
¤29
©30
ª31
·32
¸33
½34
¾35
Ã36
Ä37
²
Înon_trainable_variables
Ïlayer_metrics
"regularization_losses
#	variables
Ðlayers
$trainable_variables
 Ñlayer_regularization_losses
Òmetrics
 
\Z
VARIABLE_VALUEconv2d_57/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_57/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
²
Ónon_trainable_variables
Ôlayer_metrics
)regularization_losses
*	variables
Õlayers
+trainable_variables
 Ölayer_regularization_losses
×metrics
\Z
VARIABLE_VALUEconv2d_58/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_58/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
²
Ønon_trainable_variables
Ùlayer_metrics
/regularization_losses
0	variables
Úlayers
1trainable_variables
 Ûlayer_regularization_losses
Ümetrics
 
 
 
²
Ýnon_trainable_variables
Þlayer_metrics
3regularization_losses
4	variables
ßlayers
5trainable_variables
 àlayer_regularization_losses
ámetrics
\Z
VARIABLE_VALUEconv2d_59/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_59/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
²
ânon_trainable_variables
ãlayer_metrics
9regularization_losses
:	variables
älayers
;trainable_variables
 ålayer_regularization_losses
æmetrics
\Z
VARIABLE_VALUEconv2d_60/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_60/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
²
çnon_trainable_variables
èlayer_metrics
?regularization_losses
@	variables
élayers
Atrainable_variables
 êlayer_regularization_losses
ëmetrics
 
 
 
²
ìnon_trainable_variables
ílayer_metrics
Cregularization_losses
D	variables
îlayers
Etrainable_variables
 ïlayer_regularization_losses
ðmetrics
\Z
VARIABLE_VALUEconv2d_61/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_61/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

G0
H1
²
ñnon_trainable_variables
òlayer_metrics
Iregularization_losses
J	variables
ólayers
Ktrainable_variables
 ôlayer_regularization_losses
õmetrics
\Z
VARIABLE_VALUEconv2d_62/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_62/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1

M0
N1
²
önon_trainable_variables
÷layer_metrics
Oregularization_losses
P	variables
ølayers
Qtrainable_variables
 ùlayer_regularization_losses
úmetrics
 
 
 
²
ûnon_trainable_variables
ülayer_metrics
Sregularization_losses
T	variables
ýlayers
Utrainable_variables
 þlayer_regularization_losses
ÿmetrics
\Z
VARIABLE_VALUEconv2d_63/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_63/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

W0
X1

W0
X1
²
non_trainable_variables
layer_metrics
Yregularization_losses
Z	variables
layers
[trainable_variables
 layer_regularization_losses
metrics
\Z
VARIABLE_VALUEconv2d_64/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_64/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

]0
^1
²
non_trainable_variables
layer_metrics
_regularization_losses
`	variables
layers
atrainable_variables
 layer_regularization_losses
metrics
 
 
 
²
non_trainable_variables
layer_metrics
cregularization_losses
d	variables
layers
etrainable_variables
 layer_regularization_losses
metrics
\Z
VARIABLE_VALUEconv2d_65/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_65/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1

g0
h1
²
non_trainable_variables
layer_metrics
iregularization_losses
j	variables
layers
ktrainable_variables
 layer_regularization_losses
metrics
\Z
VARIABLE_VALUEconv2d_66/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_66/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

m0
n1
²
non_trainable_variables
layer_metrics
oregularization_losses
p	variables
layers
qtrainable_variables
 layer_regularization_losses
metrics
 
 
 
²
non_trainable_variables
layer_metrics
sregularization_losses
t	variables
layers
utrainable_variables
 layer_regularization_losses
metrics
 
 
 
²
non_trainable_variables
layer_metrics
wregularization_losses
x	variables
 layers
ytrainable_variables
 ¡layer_regularization_losses
¢metrics
][
VARIABLE_VALUEconv2d_67/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_67/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

{0
|1

{0
|1
²
£non_trainable_variables
¤layer_metrics
}regularization_losses
~	variables
¥layers
trainable_variables
 ¦layer_regularization_losses
§metrics
][
VARIABLE_VALUEconv2d_68/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_68/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
¨non_trainable_variables
©layer_metrics
regularization_losses
	variables
ªlayers
trainable_variables
 «layer_regularization_losses
¬metrics
 
 
 
µ
­non_trainable_variables
®layer_metrics
regularization_losses
	variables
¯layers
trainable_variables
 °layer_regularization_losses
±metrics
 
 
 
µ
²non_trainable_variables
³layer_metrics
regularization_losses
	variables
´layers
trainable_variables
 µlayer_regularization_losses
¶metrics
][
VARIABLE_VALUEconv2d_69/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_69/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
·non_trainable_variables
¸layer_metrics
regularization_losses
	variables
¹layers
trainable_variables
 ºlayer_regularization_losses
»metrics
][
VARIABLE_VALUEconv2d_70/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_70/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
µ
¼non_trainable_variables
½layer_metrics
regularization_losses
	variables
¾layers
trainable_variables
 ¿layer_regularization_losses
Àmetrics
 
 
 
µ
Ánon_trainable_variables
Âlayer_metrics
regularization_losses
	variables
Ãlayers
trainable_variables
 Älayer_regularization_losses
Åmetrics
 
 
 
µ
Ænon_trainable_variables
Çlayer_metrics
regularization_losses
 	variables
Èlayers
¡trainable_variables
 Élayer_regularization_losses
Êmetrics
][
VARIABLE_VALUEconv2d_71/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_71/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

£0
¤1

£0
¤1
µ
Ënon_trainable_variables
Ìlayer_metrics
¥regularization_losses
¦	variables
Ílayers
§trainable_variables
 Îlayer_regularization_losses
Ïmetrics
][
VARIABLE_VALUEconv2d_72/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_72/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

©0
ª1

©0
ª1
µ
Ðnon_trainable_variables
Ñlayer_metrics
«regularization_losses
¬	variables
Òlayers
­trainable_variables
 Ólayer_regularization_losses
Ômetrics
 
 
 
µ
Õnon_trainable_variables
Ölayer_metrics
¯regularization_losses
°	variables
×layers
±trainable_variables
 Ølayer_regularization_losses
Ùmetrics
 
 
 
µ
Únon_trainable_variables
Ûlayer_metrics
³regularization_losses
´	variables
Ülayers
µtrainable_variables
 Ýlayer_regularization_losses
Þmetrics
][
VARIABLE_VALUEconv2d_73/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_73/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

·0
¸1

·0
¸1
µ
ßnon_trainable_variables
àlayer_metrics
¹regularization_losses
º	variables
álayers
»trainable_variables
 âlayer_regularization_losses
ãmetrics
][
VARIABLE_VALUEconv2d_74/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_74/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

½0
¾1

½0
¾1
µ
änon_trainable_variables
ålayer_metrics
¿regularization_losses
À	variables
ælayers
Átrainable_variables
 çlayer_regularization_losses
èmetrics
][
VARIABLE_VALUEconv2d_75/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_75/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ã0
Ä1

Ã0
Ä1
µ
énon_trainable_variables
êlayer_metrics
Åregularization_losses
Æ	variables
ëlayers
Çtrainable_variables
 ìlayer_regularization_losses
ímetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
ö
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
 
0
î0
ï1
ð2
ñ3
ò4
ó5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

ôtotal

õcount
ö	variables
÷	keras_api
I

øtotal

ùcount
ú
_fn_kwargs
û	variables
ü	keras_api
I

ýtotal

þcount
ÿ
_fn_kwargs
	variables
	keras_api
v
true_positives
true_negatives
false_positives
false_negatives
	variables
	keras_api
\

thresholds
true_positives
false_positives
	variables
	keras_api
\

thresholds
true_positives
false_negatives
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ô0
õ1

ö	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ø0
ù1

û	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

ý0
þ1

	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3

	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
 
ca
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
}
VARIABLE_VALUEAdam/conv2d_57/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_57/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_58/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_58/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_59/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_59/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_60/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_60/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_61/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_61/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_62/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_62/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_63/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_63/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_64/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_64/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_65/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_65/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_66/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_66/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_67/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_67/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_68/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_68/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_69/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_69/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_70/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_70/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_71/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_71/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_72/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_72/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_73/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_73/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_74/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_74/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_75/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_75/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_57/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_57/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_58/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_58/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_59/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_59/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_60/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_60/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_61/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_61/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_62/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_62/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_63/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_63/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_64/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_64/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_65/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_65/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_66/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_66/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_67/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_67/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_68/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_68/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_69/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_69/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_70/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_70/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_71/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_71/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_72/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_72/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_73/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_73/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_74/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_74/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_75/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_75/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_4Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d_57/kernelconv2d_57/biasconv2d_58/kernelconv2d_58/biasconv2d_59/kernelconv2d_59/biasconv2d_60/kernelconv2d_60/biasconv2d_61/kernelconv2d_61/biasconv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/biasconv2d_64/kernelconv2d_64/biasconv2d_65/kernelconv2d_65/biasconv2d_66/kernelconv2d_66/biasconv2d_67/kernelconv2d_67/biasconv2d_68/kernelconv2d_68/biasconv2d_69/kernelconv2d_69/biasconv2d_70/kernelconv2d_70/biasconv2d_71/kernelconv2d_71/biasconv2d_72/kernelconv2d_72/biasconv2d_73/kernelconv2d_73/biasconv2d_74/kernelconv2d_74/biasconv2d_75/kernelconv2d_75/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_70430
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
³-
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_57/kernel/Read/ReadVariableOp"conv2d_57/bias/Read/ReadVariableOp$conv2d_58/kernel/Read/ReadVariableOp"conv2d_58/bias/Read/ReadVariableOp$conv2d_59/kernel/Read/ReadVariableOp"conv2d_59/bias/Read/ReadVariableOp$conv2d_60/kernel/Read/ReadVariableOp"conv2d_60/bias/Read/ReadVariableOp$conv2d_61/kernel/Read/ReadVariableOp"conv2d_61/bias/Read/ReadVariableOp$conv2d_62/kernel/Read/ReadVariableOp"conv2d_62/bias/Read/ReadVariableOp$conv2d_63/kernel/Read/ReadVariableOp"conv2d_63/bias/Read/ReadVariableOp$conv2d_64/kernel/Read/ReadVariableOp"conv2d_64/bias/Read/ReadVariableOp$conv2d_65/kernel/Read/ReadVariableOp"conv2d_65/bias/Read/ReadVariableOp$conv2d_66/kernel/Read/ReadVariableOp"conv2d_66/bias/Read/ReadVariableOp$conv2d_67/kernel/Read/ReadVariableOp"conv2d_67/bias/Read/ReadVariableOp$conv2d_68/kernel/Read/ReadVariableOp"conv2d_68/bias/Read/ReadVariableOp$conv2d_69/kernel/Read/ReadVariableOp"conv2d_69/bias/Read/ReadVariableOp$conv2d_70/kernel/Read/ReadVariableOp"conv2d_70/bias/Read/ReadVariableOp$conv2d_71/kernel/Read/ReadVariableOp"conv2d_71/bias/Read/ReadVariableOp$conv2d_72/kernel/Read/ReadVariableOp"conv2d_72/bias/Read/ReadVariableOp$conv2d_73/kernel/Read/ReadVariableOp"conv2d_73/bias/Read/ReadVariableOp$conv2d_74/kernel/Read/ReadVariableOp"conv2d_74/bias/Read/ReadVariableOp$conv2d_75/kernel/Read/ReadVariableOp"conv2d_75/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp+Adam/conv2d_57/kernel/m/Read/ReadVariableOp)Adam/conv2d_57/bias/m/Read/ReadVariableOp+Adam/conv2d_58/kernel/m/Read/ReadVariableOp)Adam/conv2d_58/bias/m/Read/ReadVariableOp+Adam/conv2d_59/kernel/m/Read/ReadVariableOp)Adam/conv2d_59/bias/m/Read/ReadVariableOp+Adam/conv2d_60/kernel/m/Read/ReadVariableOp)Adam/conv2d_60/bias/m/Read/ReadVariableOp+Adam/conv2d_61/kernel/m/Read/ReadVariableOp)Adam/conv2d_61/bias/m/Read/ReadVariableOp+Adam/conv2d_62/kernel/m/Read/ReadVariableOp)Adam/conv2d_62/bias/m/Read/ReadVariableOp+Adam/conv2d_63/kernel/m/Read/ReadVariableOp)Adam/conv2d_63/bias/m/Read/ReadVariableOp+Adam/conv2d_64/kernel/m/Read/ReadVariableOp)Adam/conv2d_64/bias/m/Read/ReadVariableOp+Adam/conv2d_65/kernel/m/Read/ReadVariableOp)Adam/conv2d_65/bias/m/Read/ReadVariableOp+Adam/conv2d_66/kernel/m/Read/ReadVariableOp)Adam/conv2d_66/bias/m/Read/ReadVariableOp+Adam/conv2d_67/kernel/m/Read/ReadVariableOp)Adam/conv2d_67/bias/m/Read/ReadVariableOp+Adam/conv2d_68/kernel/m/Read/ReadVariableOp)Adam/conv2d_68/bias/m/Read/ReadVariableOp+Adam/conv2d_69/kernel/m/Read/ReadVariableOp)Adam/conv2d_69/bias/m/Read/ReadVariableOp+Adam/conv2d_70/kernel/m/Read/ReadVariableOp)Adam/conv2d_70/bias/m/Read/ReadVariableOp+Adam/conv2d_71/kernel/m/Read/ReadVariableOp)Adam/conv2d_71/bias/m/Read/ReadVariableOp+Adam/conv2d_72/kernel/m/Read/ReadVariableOp)Adam/conv2d_72/bias/m/Read/ReadVariableOp+Adam/conv2d_73/kernel/m/Read/ReadVariableOp)Adam/conv2d_73/bias/m/Read/ReadVariableOp+Adam/conv2d_74/kernel/m/Read/ReadVariableOp)Adam/conv2d_74/bias/m/Read/ReadVariableOp+Adam/conv2d_75/kernel/m/Read/ReadVariableOp)Adam/conv2d_75/bias/m/Read/ReadVariableOp+Adam/conv2d_57/kernel/v/Read/ReadVariableOp)Adam/conv2d_57/bias/v/Read/ReadVariableOp+Adam/conv2d_58/kernel/v/Read/ReadVariableOp)Adam/conv2d_58/bias/v/Read/ReadVariableOp+Adam/conv2d_59/kernel/v/Read/ReadVariableOp)Adam/conv2d_59/bias/v/Read/ReadVariableOp+Adam/conv2d_60/kernel/v/Read/ReadVariableOp)Adam/conv2d_60/bias/v/Read/ReadVariableOp+Adam/conv2d_61/kernel/v/Read/ReadVariableOp)Adam/conv2d_61/bias/v/Read/ReadVariableOp+Adam/conv2d_62/kernel/v/Read/ReadVariableOp)Adam/conv2d_62/bias/v/Read/ReadVariableOp+Adam/conv2d_63/kernel/v/Read/ReadVariableOp)Adam/conv2d_63/bias/v/Read/ReadVariableOp+Adam/conv2d_64/kernel/v/Read/ReadVariableOp)Adam/conv2d_64/bias/v/Read/ReadVariableOp+Adam/conv2d_65/kernel/v/Read/ReadVariableOp)Adam/conv2d_65/bias/v/Read/ReadVariableOp+Adam/conv2d_66/kernel/v/Read/ReadVariableOp)Adam/conv2d_66/bias/v/Read/ReadVariableOp+Adam/conv2d_67/kernel/v/Read/ReadVariableOp)Adam/conv2d_67/bias/v/Read/ReadVariableOp+Adam/conv2d_68/kernel/v/Read/ReadVariableOp)Adam/conv2d_68/bias/v/Read/ReadVariableOp+Adam/conv2d_69/kernel/v/Read/ReadVariableOp)Adam/conv2d_69/bias/v/Read/ReadVariableOp+Adam/conv2d_70/kernel/v/Read/ReadVariableOp)Adam/conv2d_70/bias/v/Read/ReadVariableOp+Adam/conv2d_71/kernel/v/Read/ReadVariableOp)Adam/conv2d_71/bias/v/Read/ReadVariableOp+Adam/conv2d_72/kernel/v/Read/ReadVariableOp)Adam/conv2d_72/bias/v/Read/ReadVariableOp+Adam/conv2d_73/kernel/v/Read/ReadVariableOp)Adam/conv2d_73/bias/v/Read/ReadVariableOp+Adam/conv2d_74/kernel/v/Read/ReadVariableOp)Adam/conv2d_74/bias/v/Read/ReadVariableOp+Adam/conv2d_75/kernel/v/Read/ReadVariableOp)Adam/conv2d_75/bias/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_71776
Ê
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_57/kernelconv2d_57/biasconv2d_58/kernelconv2d_58/biasconv2d_59/kernelconv2d_59/biasconv2d_60/kernelconv2d_60/biasconv2d_61/kernelconv2d_61/biasconv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/biasconv2d_64/kernelconv2d_64/biasconv2d_65/kernelconv2d_65/biasconv2d_66/kernelconv2d_66/biasconv2d_67/kernelconv2d_67/biasconv2d_68/kernelconv2d_68/biasconv2d_69/kernelconv2d_69/biasconv2d_70/kernelconv2d_70/biasconv2d_71/kernelconv2d_71/biasconv2d_72/kernelconv2d_72/biasconv2d_73/kernelconv2d_73/biasconv2d_74/kernelconv2d_74/biasconv2d_75/kernelconv2d_75/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2true_positivestrue_negativesfalse_positivesfalse_negativestrue_positives_1false_positives_1true_positives_2false_negatives_1Adam/conv2d_57/kernel/mAdam/conv2d_57/bias/mAdam/conv2d_58/kernel/mAdam/conv2d_58/bias/mAdam/conv2d_59/kernel/mAdam/conv2d_59/bias/mAdam/conv2d_60/kernel/mAdam/conv2d_60/bias/mAdam/conv2d_61/kernel/mAdam/conv2d_61/bias/mAdam/conv2d_62/kernel/mAdam/conv2d_62/bias/mAdam/conv2d_63/kernel/mAdam/conv2d_63/bias/mAdam/conv2d_64/kernel/mAdam/conv2d_64/bias/mAdam/conv2d_65/kernel/mAdam/conv2d_65/bias/mAdam/conv2d_66/kernel/mAdam/conv2d_66/bias/mAdam/conv2d_67/kernel/mAdam/conv2d_67/bias/mAdam/conv2d_68/kernel/mAdam/conv2d_68/bias/mAdam/conv2d_69/kernel/mAdam/conv2d_69/bias/mAdam/conv2d_70/kernel/mAdam/conv2d_70/bias/mAdam/conv2d_71/kernel/mAdam/conv2d_71/bias/mAdam/conv2d_72/kernel/mAdam/conv2d_72/bias/mAdam/conv2d_73/kernel/mAdam/conv2d_73/bias/mAdam/conv2d_74/kernel/mAdam/conv2d_74/bias/mAdam/conv2d_75/kernel/mAdam/conv2d_75/bias/mAdam/conv2d_57/kernel/vAdam/conv2d_57/bias/vAdam/conv2d_58/kernel/vAdam/conv2d_58/bias/vAdam/conv2d_59/kernel/vAdam/conv2d_59/bias/vAdam/conv2d_60/kernel/vAdam/conv2d_60/bias/vAdam/conv2d_61/kernel/vAdam/conv2d_61/bias/vAdam/conv2d_62/kernel/vAdam/conv2d_62/bias/vAdam/conv2d_63/kernel/vAdam/conv2d_63/bias/vAdam/conv2d_64/kernel/vAdam/conv2d_64/bias/vAdam/conv2d_65/kernel/vAdam/conv2d_65/bias/vAdam/conv2d_66/kernel/vAdam/conv2d_66/bias/vAdam/conv2d_67/kernel/vAdam/conv2d_67/bias/vAdam/conv2d_68/kernel/vAdam/conv2d_68/bias/vAdam/conv2d_69/kernel/vAdam/conv2d_69/bias/vAdam/conv2d_70/kernel/vAdam/conv2d_70/bias/vAdam/conv2d_71/kernel/vAdam/conv2d_71/bias/vAdam/conv2d_72/kernel/vAdam/conv2d_72/bias/vAdam/conv2d_73/kernel/vAdam/conv2d_73/bias/vAdam/conv2d_74/kernel/vAdam/conv2d_74/bias/vAdam/conv2d_75/kernel/vAdam/conv2d_75/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_72185ýÕ
 
ý
D__inference_conv2d_74_layer_call_and_return_conditional_losses_71334

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¢
ý
D__inference_conv2d_75_layer_call_and_return_conditional_losses_69461

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ú
L
0__inference_max_pooling2d_13_layer_call_fn_68993

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_689872
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

±

'__inference_model_3_layer_call_fn_70592

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	%

unknown_27:À@

unknown_28:@$

unknown_29:@@

unknown_30:@$

unknown_31:` 

unknown_32: $

unknown_33:  

unknown_34: $

unknown_35: 

unknown_36:
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_699592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
å
B__inference_model_3_layer_call_and_return_conditional_losses_70922

inputsB
(conv2d_57_conv2d_readvariableop_resource: 7
)conv2d_57_biasadd_readvariableop_resource: B
(conv2d_58_conv2d_readvariableop_resource:  7
)conv2d_58_biasadd_readvariableop_resource: B
(conv2d_59_conv2d_readvariableop_resource: @7
)conv2d_59_biasadd_readvariableop_resource:@B
(conv2d_60_conv2d_readvariableop_resource:@@7
)conv2d_60_biasadd_readvariableop_resource:@C
(conv2d_61_conv2d_readvariableop_resource:@8
)conv2d_61_biasadd_readvariableop_resource:	D
(conv2d_62_conv2d_readvariableop_resource:8
)conv2d_62_biasadd_readvariableop_resource:	D
(conv2d_63_conv2d_readvariableop_resource:8
)conv2d_63_biasadd_readvariableop_resource:	D
(conv2d_64_conv2d_readvariableop_resource:8
)conv2d_64_biasadd_readvariableop_resource:	D
(conv2d_65_conv2d_readvariableop_resource:8
)conv2d_65_biasadd_readvariableop_resource:	D
(conv2d_66_conv2d_readvariableop_resource:8
)conv2d_66_biasadd_readvariableop_resource:	D
(conv2d_67_conv2d_readvariableop_resource:8
)conv2d_67_biasadd_readvariableop_resource:	D
(conv2d_68_conv2d_readvariableop_resource:8
)conv2d_68_biasadd_readvariableop_resource:	D
(conv2d_69_conv2d_readvariableop_resource:8
)conv2d_69_biasadd_readvariableop_resource:	D
(conv2d_70_conv2d_readvariableop_resource:8
)conv2d_70_biasadd_readvariableop_resource:	C
(conv2d_71_conv2d_readvariableop_resource:À@7
)conv2d_71_biasadd_readvariableop_resource:@B
(conv2d_72_conv2d_readvariableop_resource:@@7
)conv2d_72_biasadd_readvariableop_resource:@B
(conv2d_73_conv2d_readvariableop_resource:` 7
)conv2d_73_biasadd_readvariableop_resource: B
(conv2d_74_conv2d_readvariableop_resource:  7
)conv2d_74_biasadd_readvariableop_resource: B
(conv2d_75_conv2d_readvariableop_resource: 7
)conv2d_75_biasadd_readvariableop_resource:
identity¢ conv2d_57/BiasAdd/ReadVariableOp¢conv2d_57/Conv2D/ReadVariableOp¢ conv2d_58/BiasAdd/ReadVariableOp¢conv2d_58/Conv2D/ReadVariableOp¢ conv2d_59/BiasAdd/ReadVariableOp¢conv2d_59/Conv2D/ReadVariableOp¢ conv2d_60/BiasAdd/ReadVariableOp¢conv2d_60/Conv2D/ReadVariableOp¢ conv2d_61/BiasAdd/ReadVariableOp¢conv2d_61/Conv2D/ReadVariableOp¢ conv2d_62/BiasAdd/ReadVariableOp¢conv2d_62/Conv2D/ReadVariableOp¢ conv2d_63/BiasAdd/ReadVariableOp¢conv2d_63/Conv2D/ReadVariableOp¢ conv2d_64/BiasAdd/ReadVariableOp¢conv2d_64/Conv2D/ReadVariableOp¢ conv2d_65/BiasAdd/ReadVariableOp¢conv2d_65/Conv2D/ReadVariableOp¢ conv2d_66/BiasAdd/ReadVariableOp¢conv2d_66/Conv2D/ReadVariableOp¢ conv2d_67/BiasAdd/ReadVariableOp¢conv2d_67/Conv2D/ReadVariableOp¢ conv2d_68/BiasAdd/ReadVariableOp¢conv2d_68/Conv2D/ReadVariableOp¢ conv2d_69/BiasAdd/ReadVariableOp¢conv2d_69/Conv2D/ReadVariableOp¢ conv2d_70/BiasAdd/ReadVariableOp¢conv2d_70/Conv2D/ReadVariableOp¢ conv2d_71/BiasAdd/ReadVariableOp¢conv2d_71/Conv2D/ReadVariableOp¢ conv2d_72/BiasAdd/ReadVariableOp¢conv2d_72/Conv2D/ReadVariableOp¢ conv2d_73/BiasAdd/ReadVariableOp¢conv2d_73/Conv2D/ReadVariableOp¢ conv2d_74/BiasAdd/ReadVariableOp¢conv2d_74/Conv2D/ReadVariableOp¢ conv2d_75/BiasAdd/ReadVariableOp¢conv2d_75/Conv2D/ReadVariableOp³
conv2d_57/Conv2D/ReadVariableOpReadVariableOp(conv2d_57_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_57/Conv2D/ReadVariableOpÃ
conv2d_57/Conv2DConv2Dinputs'conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d_57/Conv2Dª
 conv2d_57/BiasAdd/ReadVariableOpReadVariableOp)conv2d_57_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_57/BiasAdd/ReadVariableOp²
conv2d_57/BiasAddBiasAddconv2d_57/Conv2D:output:0(conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_57/BiasAdd
conv2d_57/ReluReluconv2d_57/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_57/Relu³
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_58/Conv2D/ReadVariableOpÙ
conv2d_58/Conv2DConv2Dconv2d_57/Relu:activations:0'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d_58/Conv2Dª
 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_58/BiasAdd/ReadVariableOp²
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_58/BiasAdd
conv2d_58/ReluReluconv2d_58/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_58/ReluÊ
max_pooling2d_12/MaxPoolMaxPoolconv2d_58/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPool³
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_59/Conv2D/ReadVariableOpÜ
conv2d_59/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
conv2d_59/Conv2Dª
 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_59/BiasAdd/ReadVariableOp°
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_59/BiasAdd~
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_59/Relu³
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_60/Conv2D/ReadVariableOp×
conv2d_60/Conv2DConv2Dconv2d_59/Relu:activations:0'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
conv2d_60/Conv2Dª
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp°
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_60/BiasAdd~
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_60/ReluÊ
max_pooling2d_13/MaxPoolMaxPoolconv2d_60/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPool´
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_61/Conv2D/ReadVariableOpÝ
conv2d_61/Conv2DConv2D!max_pooling2d_13/MaxPool:output:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv2d_61/Conv2D«
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_61/BiasAdd/ReadVariableOp±
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_61/BiasAdd
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_61/Reluµ
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_62/Conv2D/ReadVariableOpØ
conv2d_62/Conv2DConv2Dconv2d_61/Relu:activations:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv2d_62/Conv2D«
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_62/BiasAdd/ReadVariableOp±
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_62/BiasAdd
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_62/ReluË
max_pooling2d_14/MaxPoolMaxPoolconv2d_62/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_14/MaxPoolµ
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_63/Conv2D/ReadVariableOpÝ
conv2d_63/Conv2DConv2D!max_pooling2d_14/MaxPool:output:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_63/Conv2D«
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_63/BiasAdd/ReadVariableOp±
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_63/BiasAdd
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_63/Reluµ
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_64/Conv2D/ReadVariableOpØ
conv2d_64/Conv2DConv2Dconv2d_63/Relu:activations:0'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_64/Conv2D«
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_64/BiasAdd/ReadVariableOp±
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_64/BiasAdd
conv2d_64/ReluReluconv2d_64/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_64/ReluË
max_pooling2d_15/MaxPoolMaxPoolconv2d_64/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPoolµ
conv2d_65/Conv2D/ReadVariableOpReadVariableOp(conv2d_65_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_65/Conv2D/ReadVariableOpÝ
conv2d_65/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_65/Conv2D«
 conv2d_65/BiasAdd/ReadVariableOpReadVariableOp)conv2d_65_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_65/BiasAdd/ReadVariableOp±
conv2d_65/BiasAddBiasAddconv2d_65/Conv2D:output:0(conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_65/BiasAdd
conv2d_65/ReluReluconv2d_65/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_65/Reluµ
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_66/Conv2D/ReadVariableOpØ
conv2d_66/Conv2DConv2Dconv2d_65/Relu:activations:0'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_66/Conv2D«
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_66/BiasAdd/ReadVariableOp±
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_66/BiasAdd
conv2d_66/ReluReluconv2d_66/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_66/Relu
up_sampling2d_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_12/Const
up_sampling2d_12/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_12/Const_1
up_sampling2d_12/mulMulup_sampling2d_12/Const:output:0!up_sampling2d_12/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_12/mul
-up_sampling2d_12/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_66/Relu:activations:0up_sampling2d_12/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2/
-up_sampling2d_12/resize/ResizeNearestNeighborz
concatenate_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_12/concat/axis
concatenate_12/concatConcatV2>up_sampling2d_12/resize/ResizeNearestNeighbor:resized_images:0conv2d_64/Relu:activations:0#concatenate_12/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatenate_12/concatµ
conv2d_67/Conv2D/ReadVariableOpReadVariableOp(conv2d_67_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_67/Conv2D/ReadVariableOpÚ
conv2d_67/Conv2DConv2Dconcatenate_12/concat:output:0'conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_67/Conv2D«
 conv2d_67/BiasAdd/ReadVariableOpReadVariableOp)conv2d_67_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_67/BiasAdd/ReadVariableOp±
conv2d_67/BiasAddBiasAddconv2d_67/Conv2D:output:0(conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_67/BiasAdd
conv2d_67/ReluReluconv2d_67/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_67/Reluµ
conv2d_68/Conv2D/ReadVariableOpReadVariableOp(conv2d_68_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_68/Conv2D/ReadVariableOpØ
conv2d_68/Conv2DConv2Dconv2d_67/Relu:activations:0'conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_68/Conv2D«
 conv2d_68/BiasAdd/ReadVariableOpReadVariableOp)conv2d_68_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_68/BiasAdd/ReadVariableOp±
conv2d_68/BiasAddBiasAddconv2d_68/Conv2D:output:0(conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_68/BiasAdd
conv2d_68/ReluReluconv2d_68/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_68/Relu
up_sampling2d_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_13/Const
up_sampling2d_13/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_13/Const_1
up_sampling2d_13/mulMulup_sampling2d_13/Const:output:0!up_sampling2d_13/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_13/mul
-up_sampling2d_13/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_68/Relu:activations:0up_sampling2d_13/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(2/
-up_sampling2d_13/resize/ResizeNearestNeighborz
concatenate_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_13/concat/axis
concatenate_13/concatConcatV2>up_sampling2d_13/resize/ResizeNearestNeighbor:resized_images:0conv2d_62/Relu:activations:0#concatenate_13/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
concatenate_13/concatµ
conv2d_69/Conv2D/ReadVariableOpReadVariableOp(conv2d_69_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_69/Conv2D/ReadVariableOpÚ
conv2d_69/Conv2DConv2Dconcatenate_13/concat:output:0'conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv2d_69/Conv2D«
 conv2d_69/BiasAdd/ReadVariableOpReadVariableOp)conv2d_69_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_69/BiasAdd/ReadVariableOp±
conv2d_69/BiasAddBiasAddconv2d_69/Conv2D:output:0(conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_69/BiasAdd
conv2d_69/ReluReluconv2d_69/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_69/Reluµ
conv2d_70/Conv2D/ReadVariableOpReadVariableOp(conv2d_70_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_70/Conv2D/ReadVariableOpØ
conv2d_70/Conv2DConv2Dconv2d_69/Relu:activations:0'conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv2d_70/Conv2D«
 conv2d_70/BiasAdd/ReadVariableOpReadVariableOp)conv2d_70_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_70/BiasAdd/ReadVariableOp±
conv2d_70/BiasAddBiasAddconv2d_70/Conv2D:output:0(conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_70/BiasAdd
conv2d_70/ReluReluconv2d_70/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_70/Relu
up_sampling2d_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
up_sampling2d_14/Const
up_sampling2d_14/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_14/Const_1
up_sampling2d_14/mulMulup_sampling2d_14/Const:output:0!up_sampling2d_14/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_14/mul
-up_sampling2d_14/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_70/Relu:activations:0up_sampling2d_14/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
half_pixel_centers(2/
-up_sampling2d_14/resize/ResizeNearestNeighborz
concatenate_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_14/concat/axis
concatenate_14/concatConcatV2>up_sampling2d_14/resize/ResizeNearestNeighbor:resized_images:0conv2d_60/Relu:activations:0#concatenate_14/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À2
concatenate_14/concat´
conv2d_71/Conv2D/ReadVariableOpReadVariableOp(conv2d_71_conv2d_readvariableop_resource*'
_output_shapes
:À@*
dtype02!
conv2d_71/Conv2D/ReadVariableOpÙ
conv2d_71/Conv2DConv2Dconcatenate_14/concat:output:0'conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
conv2d_71/Conv2Dª
 conv2d_71/BiasAdd/ReadVariableOpReadVariableOp)conv2d_71_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_71/BiasAdd/ReadVariableOp°
conv2d_71/BiasAddBiasAddconv2d_71/Conv2D:output:0(conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_71/BiasAdd~
conv2d_71/ReluReluconv2d_71/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_71/Relu³
conv2d_72/Conv2D/ReadVariableOpReadVariableOp(conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_72/Conv2D/ReadVariableOp×
conv2d_72/Conv2DConv2Dconv2d_71/Relu:activations:0'conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
conv2d_72/Conv2Dª
 conv2d_72/BiasAdd/ReadVariableOpReadVariableOp)conv2d_72_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_72/BiasAdd/ReadVariableOp°
conv2d_72/BiasAddBiasAddconv2d_72/Conv2D:output:0(conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_72/BiasAdd~
conv2d_72/ReluReluconv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_72/Relu
up_sampling2d_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   2
up_sampling2d_15/Const
up_sampling2d_15/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_15/Const_1
up_sampling2d_15/mulMulup_sampling2d_15/Const:output:0!up_sampling2d_15/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_15/mul
-up_sampling2d_15/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_72/Relu:activations:0up_sampling2d_15/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2/
-up_sampling2d_15/resize/ResizeNearestNeighborz
concatenate_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_15/concat/axis
concatenate_15/concatConcatV2>up_sampling2d_15/resize/ResizeNearestNeighbor:resized_images:0conv2d_58/Relu:activations:0#concatenate_15/concat/axis:output:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
concatenate_15/concat³
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02!
conv2d_73/Conv2D/ReadVariableOpÛ
conv2d_73/Conv2DConv2Dconcatenate_15/concat:output:0'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d_73/Conv2Dª
 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_73/BiasAdd/ReadVariableOp²
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_73/BiasAdd
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_73/Relu³
conv2d_74/Conv2D/ReadVariableOpReadVariableOp(conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_74/Conv2D/ReadVariableOpÙ
conv2d_74/Conv2DConv2Dconv2d_73/Relu:activations:0'conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d_74/Conv2Dª
 conv2d_74/BiasAdd/ReadVariableOpReadVariableOp)conv2d_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_74/BiasAdd/ReadVariableOp²
conv2d_74/BiasAddBiasAddconv2d_74/Conv2D:output:0(conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_74/BiasAdd
conv2d_74/ReluReluconv2d_74/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_74/Relu³
conv2d_75/Conv2D/ReadVariableOpReadVariableOp(conv2d_75_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_75/Conv2D/ReadVariableOpÙ
conv2d_75/Conv2DConv2Dconv2d_74/Relu:activations:0'conv2d_75/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_75/Conv2Dª
 conv2d_75/BiasAdd/ReadVariableOpReadVariableOp)conv2d_75_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_75/BiasAdd/ReadVariableOp²
conv2d_75/BiasAddBiasAddconv2d_75/Conv2D:output:0(conv2d_75/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_75/BiasAdd
conv2d_75/SigmoidSigmoidconv2d_75/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_75/Sigmoid
IdentityIdentityconv2d_75/Sigmoid:y:0!^conv2d_57/BiasAdd/ReadVariableOp ^conv2d_57/Conv2D/ReadVariableOp!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp!^conv2d_65/BiasAdd/ReadVariableOp ^conv2d_65/Conv2D/ReadVariableOp!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOp!^conv2d_67/BiasAdd/ReadVariableOp ^conv2d_67/Conv2D/ReadVariableOp!^conv2d_68/BiasAdd/ReadVariableOp ^conv2d_68/Conv2D/ReadVariableOp!^conv2d_69/BiasAdd/ReadVariableOp ^conv2d_69/Conv2D/ReadVariableOp!^conv2d_70/BiasAdd/ReadVariableOp ^conv2d_70/Conv2D/ReadVariableOp!^conv2d_71/BiasAdd/ReadVariableOp ^conv2d_71/Conv2D/ReadVariableOp!^conv2d_72/BiasAdd/ReadVariableOp ^conv2d_72/Conv2D/ReadVariableOp!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp!^conv2d_74/BiasAdd/ReadVariableOp ^conv2d_74/Conv2D/ReadVariableOp!^conv2d_75/BiasAdd/ReadVariableOp ^conv2d_75/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_57/BiasAdd/ReadVariableOp conv2d_57/BiasAdd/ReadVariableOp2B
conv2d_57/Conv2D/ReadVariableOpconv2d_57/Conv2D/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp2D
 conv2d_65/BiasAdd/ReadVariableOp conv2d_65/BiasAdd/ReadVariableOp2B
conv2d_65/Conv2D/ReadVariableOpconv2d_65/Conv2D/ReadVariableOp2D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2D
 conv2d_67/BiasAdd/ReadVariableOp conv2d_67/BiasAdd/ReadVariableOp2B
conv2d_67/Conv2D/ReadVariableOpconv2d_67/Conv2D/ReadVariableOp2D
 conv2d_68/BiasAdd/ReadVariableOp conv2d_68/BiasAdd/ReadVariableOp2B
conv2d_68/Conv2D/ReadVariableOpconv2d_68/Conv2D/ReadVariableOp2D
 conv2d_69/BiasAdd/ReadVariableOp conv2d_69/BiasAdd/ReadVariableOp2B
conv2d_69/Conv2D/ReadVariableOpconv2d_69/Conv2D/ReadVariableOp2D
 conv2d_70/BiasAdd/ReadVariableOp conv2d_70/BiasAdd/ReadVariableOp2B
conv2d_70/Conv2D/ReadVariableOpconv2d_70/Conv2D/ReadVariableOp2D
 conv2d_71/BiasAdd/ReadVariableOp conv2d_71/BiasAdd/ReadVariableOp2B
conv2d_71/Conv2D/ReadVariableOpconv2d_71/Conv2D/ReadVariableOp2D
 conv2d_72/BiasAdd/ReadVariableOp conv2d_72/BiasAdd/ReadVariableOp2B
conv2d_72/Conv2D/ReadVariableOpconv2d_72/Conv2D/ReadVariableOp2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp2D
 conv2d_74/BiasAdd/ReadVariableOp conv2d_74/BiasAdd/ReadVariableOp2B
conv2d_74/Conv2D/ReadVariableOpconv2d_74/Conv2D/ReadVariableOp2D
 conv2d_75/BiasAdd/ReadVariableOp conv2d_75/BiasAdd/ReadVariableOp2B
conv2d_75/Conv2D/ReadVariableOpconv2d_75/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
s
I__inference_concatenate_12_layer_call_and_return_conditional_losses_69282

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:XT
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

D__inference_conv2d_70_layer_call_and_return_conditional_losses_69356

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
 

D__inference_conv2d_69_layer_call_and_return_conditional_losses_71208

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¼
s
I__inference_concatenate_15_layer_call_and_return_conditional_losses_69414

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ :i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

ý
D__inference_conv2d_59_layer_call_and_return_conditional_losses_70982

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ 
 
_user_specified_nameinputs
Í
¡
)__inference_conv2d_67_layer_call_fn_71144

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_67_layer_call_and_return_conditional_losses_692952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
ý
D__inference_conv2d_58_layer_call_and_return_conditional_losses_69128

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
 

D__inference_conv2d_63_layer_call_and_return_conditional_losses_69216

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
ý
D__inference_conv2d_75_layer_call_and_return_conditional_losses_71354

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
«
g
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_68999

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ý
D__inference_conv2d_59_layer_call_and_return_conditional_losses_69146

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ 
 
_user_specified_nameinputs
 

D__inference_conv2d_62_layer_call_and_return_conditional_losses_71042

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
 
ý
D__inference_conv2d_74_layer_call_and_return_conditional_losses_69444

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
 

D__inference_conv2d_64_layer_call_and_return_conditional_losses_71082

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
V
!__inference__traced_restore_72185
file_prefix;
!assignvariableop_conv2d_57_kernel: /
!assignvariableop_1_conv2d_57_bias: =
#assignvariableop_2_conv2d_58_kernel:  /
!assignvariableop_3_conv2d_58_bias: =
#assignvariableop_4_conv2d_59_kernel: @/
!assignvariableop_5_conv2d_59_bias:@=
#assignvariableop_6_conv2d_60_kernel:@@/
!assignvariableop_7_conv2d_60_bias:@>
#assignvariableop_8_conv2d_61_kernel:@0
!assignvariableop_9_conv2d_61_bias:	@
$assignvariableop_10_conv2d_62_kernel:1
"assignvariableop_11_conv2d_62_bias:	@
$assignvariableop_12_conv2d_63_kernel:1
"assignvariableop_13_conv2d_63_bias:	@
$assignvariableop_14_conv2d_64_kernel:1
"assignvariableop_15_conv2d_64_bias:	@
$assignvariableop_16_conv2d_65_kernel:1
"assignvariableop_17_conv2d_65_bias:	@
$assignvariableop_18_conv2d_66_kernel:1
"assignvariableop_19_conv2d_66_bias:	@
$assignvariableop_20_conv2d_67_kernel:1
"assignvariableop_21_conv2d_67_bias:	@
$assignvariableop_22_conv2d_68_kernel:1
"assignvariableop_23_conv2d_68_bias:	@
$assignvariableop_24_conv2d_69_kernel:1
"assignvariableop_25_conv2d_69_bias:	@
$assignvariableop_26_conv2d_70_kernel:1
"assignvariableop_27_conv2d_70_bias:	?
$assignvariableop_28_conv2d_71_kernel:À@0
"assignvariableop_29_conv2d_71_bias:@>
$assignvariableop_30_conv2d_72_kernel:@@0
"assignvariableop_31_conv2d_72_bias:@>
$assignvariableop_32_conv2d_73_kernel:` 0
"assignvariableop_33_conv2d_73_bias: >
$assignvariableop_34_conv2d_74_kernel:  0
"assignvariableop_35_conv2d_74_bias: >
$assignvariableop_36_conv2d_75_kernel: 0
"assignvariableop_37_conv2d_75_bias:'
assignvariableop_38_adam_iter:	 )
assignvariableop_39_adam_beta_1: )
assignvariableop_40_adam_beta_2: (
assignvariableop_41_adam_decay: 0
&assignvariableop_42_adam_learning_rate: #
assignvariableop_43_total: #
assignvariableop_44_count: %
assignvariableop_45_total_1: %
assignvariableop_46_count_1: %
assignvariableop_47_total_2: %
assignvariableop_48_count_2: 1
"assignvariableop_49_true_positives:	È1
"assignvariableop_50_true_negatives:	È2
#assignvariableop_51_false_positives:	È2
#assignvariableop_52_false_negatives:	È2
$assignvariableop_53_true_positives_1:3
%assignvariableop_54_false_positives_1:2
$assignvariableop_55_true_positives_2:3
%assignvariableop_56_false_negatives_1:E
+assignvariableop_57_adam_conv2d_57_kernel_m: 7
)assignvariableop_58_adam_conv2d_57_bias_m: E
+assignvariableop_59_adam_conv2d_58_kernel_m:  7
)assignvariableop_60_adam_conv2d_58_bias_m: E
+assignvariableop_61_adam_conv2d_59_kernel_m: @7
)assignvariableop_62_adam_conv2d_59_bias_m:@E
+assignvariableop_63_adam_conv2d_60_kernel_m:@@7
)assignvariableop_64_adam_conv2d_60_bias_m:@F
+assignvariableop_65_adam_conv2d_61_kernel_m:@8
)assignvariableop_66_adam_conv2d_61_bias_m:	G
+assignvariableop_67_adam_conv2d_62_kernel_m:8
)assignvariableop_68_adam_conv2d_62_bias_m:	G
+assignvariableop_69_adam_conv2d_63_kernel_m:8
)assignvariableop_70_adam_conv2d_63_bias_m:	G
+assignvariableop_71_adam_conv2d_64_kernel_m:8
)assignvariableop_72_adam_conv2d_64_bias_m:	G
+assignvariableop_73_adam_conv2d_65_kernel_m:8
)assignvariableop_74_adam_conv2d_65_bias_m:	G
+assignvariableop_75_adam_conv2d_66_kernel_m:8
)assignvariableop_76_adam_conv2d_66_bias_m:	G
+assignvariableop_77_adam_conv2d_67_kernel_m:8
)assignvariableop_78_adam_conv2d_67_bias_m:	G
+assignvariableop_79_adam_conv2d_68_kernel_m:8
)assignvariableop_80_adam_conv2d_68_bias_m:	G
+assignvariableop_81_adam_conv2d_69_kernel_m:8
)assignvariableop_82_adam_conv2d_69_bias_m:	G
+assignvariableop_83_adam_conv2d_70_kernel_m:8
)assignvariableop_84_adam_conv2d_70_bias_m:	F
+assignvariableop_85_adam_conv2d_71_kernel_m:À@7
)assignvariableop_86_adam_conv2d_71_bias_m:@E
+assignvariableop_87_adam_conv2d_72_kernel_m:@@7
)assignvariableop_88_adam_conv2d_72_bias_m:@E
+assignvariableop_89_adam_conv2d_73_kernel_m:` 7
)assignvariableop_90_adam_conv2d_73_bias_m: E
+assignvariableop_91_adam_conv2d_74_kernel_m:  7
)assignvariableop_92_adam_conv2d_74_bias_m: E
+assignvariableop_93_adam_conv2d_75_kernel_m: 7
)assignvariableop_94_adam_conv2d_75_bias_m:E
+assignvariableop_95_adam_conv2d_57_kernel_v: 7
)assignvariableop_96_adam_conv2d_57_bias_v: E
+assignvariableop_97_adam_conv2d_58_kernel_v:  7
)assignvariableop_98_adam_conv2d_58_bias_v: E
+assignvariableop_99_adam_conv2d_59_kernel_v: @8
*assignvariableop_100_adam_conv2d_59_bias_v:@F
,assignvariableop_101_adam_conv2d_60_kernel_v:@@8
*assignvariableop_102_adam_conv2d_60_bias_v:@G
,assignvariableop_103_adam_conv2d_61_kernel_v:@9
*assignvariableop_104_adam_conv2d_61_bias_v:	H
,assignvariableop_105_adam_conv2d_62_kernel_v:9
*assignvariableop_106_adam_conv2d_62_bias_v:	H
,assignvariableop_107_adam_conv2d_63_kernel_v:9
*assignvariableop_108_adam_conv2d_63_bias_v:	H
,assignvariableop_109_adam_conv2d_64_kernel_v:9
*assignvariableop_110_adam_conv2d_64_bias_v:	H
,assignvariableop_111_adam_conv2d_65_kernel_v:9
*assignvariableop_112_adam_conv2d_65_bias_v:	H
,assignvariableop_113_adam_conv2d_66_kernel_v:9
*assignvariableop_114_adam_conv2d_66_bias_v:	H
,assignvariableop_115_adam_conv2d_67_kernel_v:9
*assignvariableop_116_adam_conv2d_67_bias_v:	H
,assignvariableop_117_adam_conv2d_68_kernel_v:9
*assignvariableop_118_adam_conv2d_68_bias_v:	H
,assignvariableop_119_adam_conv2d_69_kernel_v:9
*assignvariableop_120_adam_conv2d_69_bias_v:	H
,assignvariableop_121_adam_conv2d_70_kernel_v:9
*assignvariableop_122_adam_conv2d_70_bias_v:	G
,assignvariableop_123_adam_conv2d_71_kernel_v:À@8
*assignvariableop_124_adam_conv2d_71_bias_v:@F
,assignvariableop_125_adam_conv2d_72_kernel_v:@@8
*assignvariableop_126_adam_conv2d_72_bias_v:@F
,assignvariableop_127_adam_conv2d_73_kernel_v:` 8
*assignvariableop_128_adam_conv2d_73_bias_v: F
,assignvariableop_129_adam_conv2d_74_kernel_v:  8
*assignvariableop_130_adam_conv2d_74_bias_v: F
,assignvariableop_131_adam_conv2d_75_kernel_v: 8
*assignvariableop_132_adam_conv2d_75_bias_v:
identity_134¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_113¢AssignVariableOp_114¢AssignVariableOp_115¢AssignVariableOp_116¢AssignVariableOp_117¢AssignVariableOp_118¢AssignVariableOp_119¢AssignVariableOp_12¢AssignVariableOp_120¢AssignVariableOp_121¢AssignVariableOp_122¢AssignVariableOp_123¢AssignVariableOp_124¢AssignVariableOp_125¢AssignVariableOp_126¢AssignVariableOp_127¢AssignVariableOp_128¢AssignVariableOp_129¢AssignVariableOp_13¢AssignVariableOp_130¢AssignVariableOp_131¢AssignVariableOp_132¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99ÒK
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ÝJ
valueÓJBÐJB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¢
valueBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÐ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*®
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_57_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_57_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_58_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_58_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_59_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_59_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_60_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_60_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¨
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_61_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_61_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_62_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_62_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¬
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_63_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ª
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_63_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¬
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_64_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ª
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_64_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¬
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_65_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ª
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_65_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¬
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_66_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ª
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_66_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¬
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_67_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ª
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_67_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¬
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv2d_68_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ª
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv2d_68_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¬
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_69_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ª
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_69_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¬
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_70_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ª
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_70_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¬
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv2d_71_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ª
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv2d_71_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¬
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_72_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ª
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_72_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¬
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_73_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ª
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_73_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¬
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv2d_74_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ª
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv2d_74_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¬
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_75_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ª
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_75_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_38¥
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_iterIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39§
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_beta_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40§
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_beta_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¦
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_decayIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42®
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_learning_rateIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¡
AssignVariableOp_43AssignVariableOpassignvariableop_43_totalIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¡
AssignVariableOp_44AssignVariableOpassignvariableop_44_countIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45£
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46£
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47£
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_2Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48£
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_2Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49ª
AssignVariableOp_49AssignVariableOp"assignvariableop_49_true_positivesIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50ª
AssignVariableOp_50AssignVariableOp"assignvariableop_50_true_negativesIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51«
AssignVariableOp_51AssignVariableOp#assignvariableop_51_false_positivesIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52«
AssignVariableOp_52AssignVariableOp#assignvariableop_52_false_negativesIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53¬
AssignVariableOp_53AssignVariableOp$assignvariableop_53_true_positives_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54­
AssignVariableOp_54AssignVariableOp%assignvariableop_54_false_positives_1Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¬
AssignVariableOp_55AssignVariableOp$assignvariableop_55_true_positives_2Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56­
AssignVariableOp_56AssignVariableOp%assignvariableop_56_false_negatives_1Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57³
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_57_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58±
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_57_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59³
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv2d_58_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60±
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv2d_58_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61³
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_59_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62±
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_59_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63³
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv2d_60_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64±
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv2d_60_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65³
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_61_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66±
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_61_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67³
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv2d_62_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68±
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv2d_62_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69³
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_63_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70±
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_63_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71³
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv2d_64_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72±
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv2d_64_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73³
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2d_65_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74±
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2d_65_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75³
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv2d_66_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76±
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv2d_66_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77³
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv2d_67_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78±
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv2d_67_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79³
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_conv2d_68_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80±
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_conv2d_68_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81³
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_conv2d_69_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82±
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_conv2d_69_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83³
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_conv2d_70_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84±
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adam_conv2d_70_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85³
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_conv2d_71_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86±
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_conv2d_71_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87³
AssignVariableOp_87AssignVariableOp+assignvariableop_87_adam_conv2d_72_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88±
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_conv2d_72_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89³
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_conv2d_73_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90±
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_conv2d_73_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91³
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_conv2d_74_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92±
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_conv2d_74_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93³
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_conv2d_75_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94±
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_conv2d_75_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95³
AssignVariableOp_95AssignVariableOp+assignvariableop_95_adam_conv2d_57_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96±
AssignVariableOp_96AssignVariableOp)assignvariableop_96_adam_conv2d_57_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97³
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_conv2d_58_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98±
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_conv2d_58_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99³
AssignVariableOp_99AssignVariableOp+assignvariableop_99_adam_conv2d_59_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100µ
AssignVariableOp_100AssignVariableOp*assignvariableop_100_adam_conv2d_59_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101·
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_conv2d_60_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102µ
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_conv2d_60_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103·
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_conv2d_61_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104µ
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_conv2d_61_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105·
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_conv2d_62_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106µ
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_conv2d_62_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107·
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_conv2d_63_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108µ
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_conv2d_63_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109·
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_conv2d_64_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110µ
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_conv2d_64_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111·
AssignVariableOp_111AssignVariableOp,assignvariableop_111_adam_conv2d_65_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112µ
AssignVariableOp_112AssignVariableOp*assignvariableop_112_adam_conv2d_65_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113·
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_conv2d_66_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114µ
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_conv2d_66_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115·
AssignVariableOp_115AssignVariableOp,assignvariableop_115_adam_conv2d_67_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116µ
AssignVariableOp_116AssignVariableOp*assignvariableop_116_adam_conv2d_67_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117·
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_conv2d_68_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118µ
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_conv2d_68_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119·
AssignVariableOp_119AssignVariableOp,assignvariableop_119_adam_conv2d_69_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120µ
AssignVariableOp_120AssignVariableOp*assignvariableop_120_adam_conv2d_69_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121·
AssignVariableOp_121AssignVariableOp,assignvariableop_121_adam_conv2d_70_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122µ
AssignVariableOp_122AssignVariableOp*assignvariableop_122_adam_conv2d_70_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123·
AssignVariableOp_123AssignVariableOp,assignvariableop_123_adam_conv2d_71_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124µ
AssignVariableOp_124AssignVariableOp*assignvariableop_124_adam_conv2d_71_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125·
AssignVariableOp_125AssignVariableOp,assignvariableop_125_adam_conv2d_72_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126µ
AssignVariableOp_126AssignVariableOp*assignvariableop_126_adam_conv2d_72_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127·
AssignVariableOp_127AssignVariableOp,assignvariableop_127_adam_conv2d_73_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128µ
AssignVariableOp_128AssignVariableOp*assignvariableop_128_adam_conv2d_73_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129·
AssignVariableOp_129AssignVariableOp,assignvariableop_129_adam_conv2d_74_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130µ
AssignVariableOp_130AssignVariableOp*assignvariableop_130_adam_conv2d_74_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131·
AssignVariableOp_131AssignVariableOp,assignvariableop_131_adam_conv2d_75_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132µ
AssignVariableOp_132AssignVariableOp*assignvariableop_132_adam_conv2d_75_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpï
Identity_133Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_133ã
Identity_134IdentityIdentity_133:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_134"%
identity_134Identity_134:output:0*¡
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Þ
ª
B__inference_model_3_layer_call_and_return_conditional_losses_70341
input_4)
conv2d_57_70233: 
conv2d_57_70235: )
conv2d_58_70238:  
conv2d_58_70240: )
conv2d_59_70244: @
conv2d_59_70246:@)
conv2d_60_70249:@@
conv2d_60_70251:@*
conv2d_61_70255:@
conv2d_61_70257:	+
conv2d_62_70260:
conv2d_62_70262:	+
conv2d_63_70266:
conv2d_63_70268:	+
conv2d_64_70271:
conv2d_64_70273:	+
conv2d_65_70277:
conv2d_65_70279:	+
conv2d_66_70282:
conv2d_66_70284:	+
conv2d_67_70289:
conv2d_67_70291:	+
conv2d_68_70294:
conv2d_68_70296:	+
conv2d_69_70301:
conv2d_69_70303:	+
conv2d_70_70306:
conv2d_70_70308:	*
conv2d_71_70313:À@
conv2d_71_70315:@)
conv2d_72_70318:@@
conv2d_72_70320:@)
conv2d_73_70325:` 
conv2d_73_70327: )
conv2d_74_70330:  
conv2d_74_70332: )
conv2d_75_70335: 
conv2d_75_70337:
identity¢!conv2d_57/StatefulPartitionedCall¢!conv2d_58/StatefulPartitionedCall¢!conv2d_59/StatefulPartitionedCall¢!conv2d_60/StatefulPartitionedCall¢!conv2d_61/StatefulPartitionedCall¢!conv2d_62/StatefulPartitionedCall¢!conv2d_63/StatefulPartitionedCall¢!conv2d_64/StatefulPartitionedCall¢!conv2d_65/StatefulPartitionedCall¢!conv2d_66/StatefulPartitionedCall¢!conv2d_67/StatefulPartitionedCall¢!conv2d_68/StatefulPartitionedCall¢!conv2d_69/StatefulPartitionedCall¢!conv2d_70/StatefulPartitionedCall¢!conv2d_71/StatefulPartitionedCall¢!conv2d_72/StatefulPartitionedCall¢!conv2d_73/StatefulPartitionedCall¢!conv2d_74/StatefulPartitionedCall¢!conv2d_75/StatefulPartitionedCall¤
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_57_70233conv2d_57_70235*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_57_layer_call_and_return_conditional_losses_691112#
!conv2d_57/StatefulPartitionedCallÇ
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0conv2d_58_70238conv2d_58_70240*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_58_layer_call_and_return_conditional_losses_691282#
!conv2d_58/StatefulPartitionedCall
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_689752"
 max_pooling2d_12/PartitionedCallÄ
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_59_70244conv2d_59_70246*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_59_layer_call_and_return_conditional_losses_691462#
!conv2d_59/StatefulPartitionedCallÅ
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0conv2d_60_70249conv2d_60_70251*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_60_layer_call_and_return_conditional_losses_691632#
!conv2d_60/StatefulPartitionedCall
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_689872"
 max_pooling2d_13/PartitionedCallÅ
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0conv2d_61_70255conv2d_61_70257*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_61_layer_call_and_return_conditional_losses_691812#
!conv2d_61/StatefulPartitionedCallÆ
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0conv2d_62_70260conv2d_62_70262*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_691982#
!conv2d_62/StatefulPartitionedCall
 max_pooling2d_14/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_689992"
 max_pooling2d_14/PartitionedCallÅ
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0conv2d_63_70266conv2d_63_70268*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_692162#
!conv2d_63/StatefulPartitionedCallÆ
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0conv2d_64_70271conv2d_64_70273*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_692332#
!conv2d_64/StatefulPartitionedCall
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_690112"
 max_pooling2d_15/PartitionedCallÅ
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_65_70277conv2d_65_70279*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_65_layer_call_and_return_conditional_losses_692512#
!conv2d_65/StatefulPartitionedCallÆ
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0conv2d_66_70282conv2d_66_70284*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_66_layer_call_and_return_conditional_losses_692682#
!conv2d_66/StatefulPartitionedCall­
 up_sampling2d_12/PartitionedCallPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_690302"
 up_sampling2d_12/PartitionedCallÁ
concatenate_12/PartitionedCallPartitionedCall)up_sampling2d_12/PartitionedCall:output:0*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_12_layer_call_and_return_conditional_losses_692822 
concatenate_12/PartitionedCallÃ
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0conv2d_67_70289conv2d_67_70291*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_67_layer_call_and_return_conditional_losses_692952#
!conv2d_67/StatefulPartitionedCallÆ
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0conv2d_68_70294conv2d_68_70296*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_68_layer_call_and_return_conditional_losses_693122#
!conv2d_68/StatefulPartitionedCall­
 up_sampling2d_13/PartitionedCallPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_690492"
 up_sampling2d_13/PartitionedCallÁ
concatenate_13/PartitionedCallPartitionedCall)up_sampling2d_13/PartitionedCall:output:0*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_13_layer_call_and_return_conditional_losses_693262 
concatenate_13/PartitionedCallÃ
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0conv2d_69_70301conv2d_69_70303*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_69_layer_call_and_return_conditional_losses_693392#
!conv2d_69/StatefulPartitionedCallÆ
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0conv2d_70_70306conv2d_70_70308*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_70_layer_call_and_return_conditional_losses_693562#
!conv2d_70/StatefulPartitionedCall­
 up_sampling2d_14/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_690682"
 up_sampling2d_14/PartitionedCallÁ
concatenate_14/PartitionedCallPartitionedCall)up_sampling2d_14/PartitionedCall:output:0*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_14_layer_call_and_return_conditional_losses_693702 
concatenate_14/PartitionedCallÂ
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0conv2d_71_70313conv2d_71_70315*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_71_layer_call_and_return_conditional_losses_693832#
!conv2d_71/StatefulPartitionedCallÅ
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0conv2d_72_70318conv2d_72_70320*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_72_layer_call_and_return_conditional_losses_694002#
!conv2d_72/StatefulPartitionedCall¬
 up_sampling2d_15/PartitionedCallPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_15_layer_call_and_return_conditional_losses_690872"
 up_sampling2d_15/PartitionedCallÂ
concatenate_15/PartitionedCallPartitionedCall)up_sampling2d_15/PartitionedCall:output:0*conv2d_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_15_layer_call_and_return_conditional_losses_694142 
concatenate_15/PartitionedCallÄ
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0conv2d_73_70325conv2d_73_70327*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_73_layer_call_and_return_conditional_losses_694272#
!conv2d_73/StatefulPartitionedCallÇ
!conv2d_74/StatefulPartitionedCallStatefulPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0conv2d_74_70330conv2d_74_70332*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_74_layer_call_and_return_conditional_losses_694442#
!conv2d_74/StatefulPartitionedCallÇ
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall*conv2d_74/StatefulPartitionedCall:output:0conv2d_75_70335conv2d_75_70337*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_75_layer_call_and_return_conditional_losses_694612#
!conv2d_75/StatefulPartitionedCall´
IdentityIdentity*conv2d_75/StatefulPartitionedCall:output:0"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall"^conv2d_74/StatefulPartitionedCall"^conv2d_75/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2F
!conv2d_74/StatefulPartitionedCall!conv2d_74/StatefulPartitionedCall2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
º
s
I__inference_concatenate_13_layer_call_and_return_conditional_losses_69326

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ  :j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:XT
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Í
¡
)__inference_conv2d_66_layer_call_fn_71111

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_66_layer_call_and_return_conditional_losses_692682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
¡
)__inference_conv2d_70_layer_call_fn_71217

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_70_layer_call_and_return_conditional_losses_693562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Î

)__inference_conv2d_74_layer_call_fn_71323

inputs!
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_74_layer_call_and_return_conditional_losses_694442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Î

)__inference_conv2d_58_layer_call_fn_70951

inputs!
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_58_layer_call_and_return_conditional_losses_691282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

ý
D__inference_conv2d_72_layer_call_and_return_conditional_losses_69400

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs

ý
D__inference_conv2d_60_layer_call_and_return_conditional_losses_71002

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Ú
L
0__inference_max_pooling2d_15_layer_call_fn_69017

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_690112
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

D__inference_conv2d_63_layer_call_and_return_conditional_losses_71062

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¯6
__inference__traced_save_71776
file_prefix/
+savev2_conv2d_57_kernel_read_readvariableop-
)savev2_conv2d_57_bias_read_readvariableop/
+savev2_conv2d_58_kernel_read_readvariableop-
)savev2_conv2d_58_bias_read_readvariableop/
+savev2_conv2d_59_kernel_read_readvariableop-
)savev2_conv2d_59_bias_read_readvariableop/
+savev2_conv2d_60_kernel_read_readvariableop-
)savev2_conv2d_60_bias_read_readvariableop/
+savev2_conv2d_61_kernel_read_readvariableop-
)savev2_conv2d_61_bias_read_readvariableop/
+savev2_conv2d_62_kernel_read_readvariableop-
)savev2_conv2d_62_bias_read_readvariableop/
+savev2_conv2d_63_kernel_read_readvariableop-
)savev2_conv2d_63_bias_read_readvariableop/
+savev2_conv2d_64_kernel_read_readvariableop-
)savev2_conv2d_64_bias_read_readvariableop/
+savev2_conv2d_65_kernel_read_readvariableop-
)savev2_conv2d_65_bias_read_readvariableop/
+savev2_conv2d_66_kernel_read_readvariableop-
)savev2_conv2d_66_bias_read_readvariableop/
+savev2_conv2d_67_kernel_read_readvariableop-
)savev2_conv2d_67_bias_read_readvariableop/
+savev2_conv2d_68_kernel_read_readvariableop-
)savev2_conv2d_68_bias_read_readvariableop/
+savev2_conv2d_69_kernel_read_readvariableop-
)savev2_conv2d_69_bias_read_readvariableop/
+savev2_conv2d_70_kernel_read_readvariableop-
)savev2_conv2d_70_bias_read_readvariableop/
+savev2_conv2d_71_kernel_read_readvariableop-
)savev2_conv2d_71_bias_read_readvariableop/
+savev2_conv2d_72_kernel_read_readvariableop-
)savev2_conv2d_72_bias_read_readvariableop/
+savev2_conv2d_73_kernel_read_readvariableop-
)savev2_conv2d_73_bias_read_readvariableop/
+savev2_conv2d_74_kernel_read_readvariableop-
)savev2_conv2d_74_bias_read_readvariableop/
+savev2_conv2d_75_kernel_read_readvariableop-
)savev2_conv2d_75_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop0
,savev2_false_positives_1_read_readvariableop/
+savev2_true_positives_2_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop6
2savev2_adam_conv2d_57_kernel_m_read_readvariableop4
0savev2_adam_conv2d_57_bias_m_read_readvariableop6
2savev2_adam_conv2d_58_kernel_m_read_readvariableop4
0savev2_adam_conv2d_58_bias_m_read_readvariableop6
2savev2_adam_conv2d_59_kernel_m_read_readvariableop4
0savev2_adam_conv2d_59_bias_m_read_readvariableop6
2savev2_adam_conv2d_60_kernel_m_read_readvariableop4
0savev2_adam_conv2d_60_bias_m_read_readvariableop6
2savev2_adam_conv2d_61_kernel_m_read_readvariableop4
0savev2_adam_conv2d_61_bias_m_read_readvariableop6
2savev2_adam_conv2d_62_kernel_m_read_readvariableop4
0savev2_adam_conv2d_62_bias_m_read_readvariableop6
2savev2_adam_conv2d_63_kernel_m_read_readvariableop4
0savev2_adam_conv2d_63_bias_m_read_readvariableop6
2savev2_adam_conv2d_64_kernel_m_read_readvariableop4
0savev2_adam_conv2d_64_bias_m_read_readvariableop6
2savev2_adam_conv2d_65_kernel_m_read_readvariableop4
0savev2_adam_conv2d_65_bias_m_read_readvariableop6
2savev2_adam_conv2d_66_kernel_m_read_readvariableop4
0savev2_adam_conv2d_66_bias_m_read_readvariableop6
2savev2_adam_conv2d_67_kernel_m_read_readvariableop4
0savev2_adam_conv2d_67_bias_m_read_readvariableop6
2savev2_adam_conv2d_68_kernel_m_read_readvariableop4
0savev2_adam_conv2d_68_bias_m_read_readvariableop6
2savev2_adam_conv2d_69_kernel_m_read_readvariableop4
0savev2_adam_conv2d_69_bias_m_read_readvariableop6
2savev2_adam_conv2d_70_kernel_m_read_readvariableop4
0savev2_adam_conv2d_70_bias_m_read_readvariableop6
2savev2_adam_conv2d_71_kernel_m_read_readvariableop4
0savev2_adam_conv2d_71_bias_m_read_readvariableop6
2savev2_adam_conv2d_72_kernel_m_read_readvariableop4
0savev2_adam_conv2d_72_bias_m_read_readvariableop6
2savev2_adam_conv2d_73_kernel_m_read_readvariableop4
0savev2_adam_conv2d_73_bias_m_read_readvariableop6
2savev2_adam_conv2d_74_kernel_m_read_readvariableop4
0savev2_adam_conv2d_74_bias_m_read_readvariableop6
2savev2_adam_conv2d_75_kernel_m_read_readvariableop4
0savev2_adam_conv2d_75_bias_m_read_readvariableop6
2savev2_adam_conv2d_57_kernel_v_read_readvariableop4
0savev2_adam_conv2d_57_bias_v_read_readvariableop6
2savev2_adam_conv2d_58_kernel_v_read_readvariableop4
0savev2_adam_conv2d_58_bias_v_read_readvariableop6
2savev2_adam_conv2d_59_kernel_v_read_readvariableop4
0savev2_adam_conv2d_59_bias_v_read_readvariableop6
2savev2_adam_conv2d_60_kernel_v_read_readvariableop4
0savev2_adam_conv2d_60_bias_v_read_readvariableop6
2savev2_adam_conv2d_61_kernel_v_read_readvariableop4
0savev2_adam_conv2d_61_bias_v_read_readvariableop6
2savev2_adam_conv2d_62_kernel_v_read_readvariableop4
0savev2_adam_conv2d_62_bias_v_read_readvariableop6
2savev2_adam_conv2d_63_kernel_v_read_readvariableop4
0savev2_adam_conv2d_63_bias_v_read_readvariableop6
2savev2_adam_conv2d_64_kernel_v_read_readvariableop4
0savev2_adam_conv2d_64_bias_v_read_readvariableop6
2savev2_adam_conv2d_65_kernel_v_read_readvariableop4
0savev2_adam_conv2d_65_bias_v_read_readvariableop6
2savev2_adam_conv2d_66_kernel_v_read_readvariableop4
0savev2_adam_conv2d_66_bias_v_read_readvariableop6
2savev2_adam_conv2d_67_kernel_v_read_readvariableop4
0savev2_adam_conv2d_67_bias_v_read_readvariableop6
2savev2_adam_conv2d_68_kernel_v_read_readvariableop4
0savev2_adam_conv2d_68_bias_v_read_readvariableop6
2savev2_adam_conv2d_69_kernel_v_read_readvariableop4
0savev2_adam_conv2d_69_bias_v_read_readvariableop6
2savev2_adam_conv2d_70_kernel_v_read_readvariableop4
0savev2_adam_conv2d_70_bias_v_read_readvariableop6
2savev2_adam_conv2d_71_kernel_v_read_readvariableop4
0savev2_adam_conv2d_71_bias_v_read_readvariableop6
2savev2_adam_conv2d_72_kernel_v_read_readvariableop4
0savev2_adam_conv2d_72_bias_v_read_readvariableop6
2savev2_adam_conv2d_73_kernel_v_read_readvariableop4
0savev2_adam_conv2d_73_bias_v_read_readvariableop6
2savev2_adam_conv2d_74_kernel_v_read_readvariableop4
0savev2_adam_conv2d_74_bias_v_read_readvariableop6
2savev2_adam_conv2d_75_kernel_v_read_readvariableop4
0savev2_adam_conv2d_75_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÌK
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ÝJ
valueÓJBÐJB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¢
valueBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesó3
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_57_kernel_read_readvariableop)savev2_conv2d_57_bias_read_readvariableop+savev2_conv2d_58_kernel_read_readvariableop)savev2_conv2d_58_bias_read_readvariableop+savev2_conv2d_59_kernel_read_readvariableop)savev2_conv2d_59_bias_read_readvariableop+savev2_conv2d_60_kernel_read_readvariableop)savev2_conv2d_60_bias_read_readvariableop+savev2_conv2d_61_kernel_read_readvariableop)savev2_conv2d_61_bias_read_readvariableop+savev2_conv2d_62_kernel_read_readvariableop)savev2_conv2d_62_bias_read_readvariableop+savev2_conv2d_63_kernel_read_readvariableop)savev2_conv2d_63_bias_read_readvariableop+savev2_conv2d_64_kernel_read_readvariableop)savev2_conv2d_64_bias_read_readvariableop+savev2_conv2d_65_kernel_read_readvariableop)savev2_conv2d_65_bias_read_readvariableop+savev2_conv2d_66_kernel_read_readvariableop)savev2_conv2d_66_bias_read_readvariableop+savev2_conv2d_67_kernel_read_readvariableop)savev2_conv2d_67_bias_read_readvariableop+savev2_conv2d_68_kernel_read_readvariableop)savev2_conv2d_68_bias_read_readvariableop+savev2_conv2d_69_kernel_read_readvariableop)savev2_conv2d_69_bias_read_readvariableop+savev2_conv2d_70_kernel_read_readvariableop)savev2_conv2d_70_bias_read_readvariableop+savev2_conv2d_71_kernel_read_readvariableop)savev2_conv2d_71_bias_read_readvariableop+savev2_conv2d_72_kernel_read_readvariableop)savev2_conv2d_72_bias_read_readvariableop+savev2_conv2d_73_kernel_read_readvariableop)savev2_conv2d_73_bias_read_readvariableop+savev2_conv2d_74_kernel_read_readvariableop)savev2_conv2d_74_bias_read_readvariableop+savev2_conv2d_75_kernel_read_readvariableop)savev2_conv2d_75_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop,savev2_false_positives_1_read_readvariableop+savev2_true_positives_2_read_readvariableop,savev2_false_negatives_1_read_readvariableop2savev2_adam_conv2d_57_kernel_m_read_readvariableop0savev2_adam_conv2d_57_bias_m_read_readvariableop2savev2_adam_conv2d_58_kernel_m_read_readvariableop0savev2_adam_conv2d_58_bias_m_read_readvariableop2savev2_adam_conv2d_59_kernel_m_read_readvariableop0savev2_adam_conv2d_59_bias_m_read_readvariableop2savev2_adam_conv2d_60_kernel_m_read_readvariableop0savev2_adam_conv2d_60_bias_m_read_readvariableop2savev2_adam_conv2d_61_kernel_m_read_readvariableop0savev2_adam_conv2d_61_bias_m_read_readvariableop2savev2_adam_conv2d_62_kernel_m_read_readvariableop0savev2_adam_conv2d_62_bias_m_read_readvariableop2savev2_adam_conv2d_63_kernel_m_read_readvariableop0savev2_adam_conv2d_63_bias_m_read_readvariableop2savev2_adam_conv2d_64_kernel_m_read_readvariableop0savev2_adam_conv2d_64_bias_m_read_readvariableop2savev2_adam_conv2d_65_kernel_m_read_readvariableop0savev2_adam_conv2d_65_bias_m_read_readvariableop2savev2_adam_conv2d_66_kernel_m_read_readvariableop0savev2_adam_conv2d_66_bias_m_read_readvariableop2savev2_adam_conv2d_67_kernel_m_read_readvariableop0savev2_adam_conv2d_67_bias_m_read_readvariableop2savev2_adam_conv2d_68_kernel_m_read_readvariableop0savev2_adam_conv2d_68_bias_m_read_readvariableop2savev2_adam_conv2d_69_kernel_m_read_readvariableop0savev2_adam_conv2d_69_bias_m_read_readvariableop2savev2_adam_conv2d_70_kernel_m_read_readvariableop0savev2_adam_conv2d_70_bias_m_read_readvariableop2savev2_adam_conv2d_71_kernel_m_read_readvariableop0savev2_adam_conv2d_71_bias_m_read_readvariableop2savev2_adam_conv2d_72_kernel_m_read_readvariableop0savev2_adam_conv2d_72_bias_m_read_readvariableop2savev2_adam_conv2d_73_kernel_m_read_readvariableop0savev2_adam_conv2d_73_bias_m_read_readvariableop2savev2_adam_conv2d_74_kernel_m_read_readvariableop0savev2_adam_conv2d_74_bias_m_read_readvariableop2savev2_adam_conv2d_75_kernel_m_read_readvariableop0savev2_adam_conv2d_75_bias_m_read_readvariableop2savev2_adam_conv2d_57_kernel_v_read_readvariableop0savev2_adam_conv2d_57_bias_v_read_readvariableop2savev2_adam_conv2d_58_kernel_v_read_readvariableop0savev2_adam_conv2d_58_bias_v_read_readvariableop2savev2_adam_conv2d_59_kernel_v_read_readvariableop0savev2_adam_conv2d_59_bias_v_read_readvariableop2savev2_adam_conv2d_60_kernel_v_read_readvariableop0savev2_adam_conv2d_60_bias_v_read_readvariableop2savev2_adam_conv2d_61_kernel_v_read_readvariableop0savev2_adam_conv2d_61_bias_v_read_readvariableop2savev2_adam_conv2d_62_kernel_v_read_readvariableop0savev2_adam_conv2d_62_bias_v_read_readvariableop2savev2_adam_conv2d_63_kernel_v_read_readvariableop0savev2_adam_conv2d_63_bias_v_read_readvariableop2savev2_adam_conv2d_64_kernel_v_read_readvariableop0savev2_adam_conv2d_64_bias_v_read_readvariableop2savev2_adam_conv2d_65_kernel_v_read_readvariableop0savev2_adam_conv2d_65_bias_v_read_readvariableop2savev2_adam_conv2d_66_kernel_v_read_readvariableop0savev2_adam_conv2d_66_bias_v_read_readvariableop2savev2_adam_conv2d_67_kernel_v_read_readvariableop0savev2_adam_conv2d_67_bias_v_read_readvariableop2savev2_adam_conv2d_68_kernel_v_read_readvariableop0savev2_adam_conv2d_68_bias_v_read_readvariableop2savev2_adam_conv2d_69_kernel_v_read_readvariableop0savev2_adam_conv2d_69_bias_v_read_readvariableop2savev2_adam_conv2d_70_kernel_v_read_readvariableop0savev2_adam_conv2d_70_bias_v_read_readvariableop2savev2_adam_conv2d_71_kernel_v_read_readvariableop0savev2_adam_conv2d_71_bias_v_read_readvariableop2savev2_adam_conv2d_72_kernel_v_read_readvariableop0savev2_adam_conv2d_72_bias_v_read_readvariableop2savev2_adam_conv2d_73_kernel_v_read_readvariableop0savev2_adam_conv2d_73_bias_v_read_readvariableop2savev2_adam_conv2d_74_kernel_v_read_readvariableop0savev2_adam_conv2d_74_bias_v_read_readvariableop2savev2_adam_conv2d_75_kernel_v_read_readvariableop0savev2_adam_conv2d_75_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : :  : : @:@:@@:@:@::::::::::::::::::::À@:@:@@:@:` : :  : : :: : : : : : : : : : : :È:È:È:È::::: : :  : : @:@:@@:@:@::::::::::::::::::::À@:@:@@:@:` : :  : : :: : :  : : @:@:@@:@:@::::::::::::::::::::À@:@:@@:@:` : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-	)
'
_output_shapes
:@:!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::-)
'
_output_shapes
:À@: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@:,!(
&
_output_shapes
:` : "

_output_shapes
: :,#(
&
_output_shapes
:  : $

_output_shapes
: :,%(
&
_output_shapes
: : &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :!2

_output_shapes	
:È:!3

_output_shapes	
:È:!4

_output_shapes	
:È:!5

_output_shapes	
:È: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
: : ;

_output_shapes
: :,<(
&
_output_shapes
:  : =

_output_shapes
: :,>(
&
_output_shapes
: @: ?

_output_shapes
:@:,@(
&
_output_shapes
:@@: A

_output_shapes
:@:-B)
'
_output_shapes
:@:!C

_output_shapes	
::.D*
(
_output_shapes
::!E

_output_shapes	
::.F*
(
_output_shapes
::!G

_output_shapes	
::.H*
(
_output_shapes
::!I

_output_shapes	
::.J*
(
_output_shapes
::!K

_output_shapes	
::.L*
(
_output_shapes
::!M

_output_shapes	
::.N*
(
_output_shapes
::!O

_output_shapes	
::.P*
(
_output_shapes
::!Q

_output_shapes	
::.R*
(
_output_shapes
::!S

_output_shapes	
::.T*
(
_output_shapes
::!U

_output_shapes	
::-V)
'
_output_shapes
:À@: W

_output_shapes
:@:,X(
&
_output_shapes
:@@: Y

_output_shapes
:@:,Z(
&
_output_shapes
:` : [

_output_shapes
: :,\(
&
_output_shapes
:  : ]

_output_shapes
: :,^(
&
_output_shapes
: : _

_output_shapes
::,`(
&
_output_shapes
: : a

_output_shapes
: :,b(
&
_output_shapes
:  : c

_output_shapes
: :,d(
&
_output_shapes
: @: e

_output_shapes
:@:,f(
&
_output_shapes
:@@: g

_output_shapes
:@:-h)
'
_output_shapes
:@:!i

_output_shapes	
::.j*
(
_output_shapes
::!k

_output_shapes	
::.l*
(
_output_shapes
::!m

_output_shapes	
::.n*
(
_output_shapes
::!o

_output_shapes	
::.p*
(
_output_shapes
::!q

_output_shapes	
::.r*
(
_output_shapes
::!s

_output_shapes	
::.t*
(
_output_shapes
::!u

_output_shapes	
::.v*
(
_output_shapes
::!w

_output_shapes	
::.x*
(
_output_shapes
::!y

_output_shapes	
::.z*
(
_output_shapes
::!{

_output_shapes	
::-|)
'
_output_shapes
:À@: }

_output_shapes
:@:,~(
&
_output_shapes
:@@: 

_output_shapes
:@:-(
&
_output_shapes
:` :!

_output_shapes
: :-(
&
_output_shapes
:  :!

_output_shapes
: :-(
&
_output_shapes
: :!

_output_shapes
::

_output_shapes
: 
¨
Z
.__inference_concatenate_13_layer_call_fn_71181
inputs_0
inputs_1
identityà
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_13_layer_call_and_return_conditional_losses_693262
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ  :l h
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/1
²
g
K__inference_up_sampling2d_15_layer_call_and_return_conditional_losses_69087

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
 
)__inference_conv2d_61_layer_call_fn_71011

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_61_layer_call_and_return_conditional_losses_691812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @
 
_user_specified_nameinputs
Î

)__inference_conv2d_57_layer_call_fn_70931

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_57_layer_call_and_return_conditional_losses_691112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

²

'__inference_model_3_layer_call_fn_69547
input_4!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	%

unknown_27:À@

unknown_28:@$

unknown_29:@@

unknown_30:@$

unknown_31:` 

unknown_32: $

unknown_33:  

unknown_34: $

unknown_35: 

unknown_36:
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_694682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
 

D__inference_conv2d_64_layer_call_and_return_conditional_losses_69233

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
ý
D__inference_conv2d_58_layer_call_and_return_conditional_losses_70962

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Â
u
I__inference_concatenate_12_layer_call_and_return_conditional_losses_71135
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:l h
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ú
L
0__inference_up_sampling2d_13_layer_call_fn_69055

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_690492
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
¡
)__inference_conv2d_65_layer_call_fn_71091

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_65_layer_call_and_return_conditional_losses_692512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
L
0__inference_up_sampling2d_12_layer_call_fn_69036

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_690302
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

D__inference_conv2d_62_layer_call_and_return_conditional_losses_69198

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
 

D__inference_conv2d_65_layer_call_and_return_conditional_losses_71102

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

þ
D__inference_conv2d_71_layer_call_and_return_conditional_losses_69383

inputs9
conv2d_readvariableop_resource:À@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:À@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@À: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À
 
_user_specified_nameinputs
 

D__inference_conv2d_66_layer_call_and_return_conditional_losses_71122

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
Z
.__inference_concatenate_12_layer_call_fn_71128
inputs_0
inputs_1
identityà
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_12_layer_call_and_return_conditional_losses_692822
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:l h
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
 

D__inference_conv2d_67_layer_call_and_return_conditional_losses_69295

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
s
I__inference_concatenate_14_layer_call_and_return_conditional_losses_69370

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@@:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs

±

'__inference_model_3_layer_call_fn_70511

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	%

unknown_27:À@

unknown_28:@$

unknown_29:@@

unknown_30:@$

unknown_31:` 

unknown_32: $

unknown_33:  

unknown_34: $

unknown_35: 

unknown_36:
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_694682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
g
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_69011

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

D__inference_conv2d_66_layer_call_and_return_conditional_losses_69268

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

D__inference_conv2d_68_layer_call_and_return_conditional_losses_71175

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
L
0__inference_up_sampling2d_14_layer_call_fn_69074

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_690682
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
L
0__inference_up_sampling2d_15_layer_call_fn_69093

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_15_layer_call_and_return_conditional_losses_690872
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
©
B__inference_model_3_layer_call_and_return_conditional_losses_69468

inputs)
conv2d_57_69112: 
conv2d_57_69114: )
conv2d_58_69129:  
conv2d_58_69131: )
conv2d_59_69147: @
conv2d_59_69149:@)
conv2d_60_69164:@@
conv2d_60_69166:@*
conv2d_61_69182:@
conv2d_61_69184:	+
conv2d_62_69199:
conv2d_62_69201:	+
conv2d_63_69217:
conv2d_63_69219:	+
conv2d_64_69234:
conv2d_64_69236:	+
conv2d_65_69252:
conv2d_65_69254:	+
conv2d_66_69269:
conv2d_66_69271:	+
conv2d_67_69296:
conv2d_67_69298:	+
conv2d_68_69313:
conv2d_68_69315:	+
conv2d_69_69340:
conv2d_69_69342:	+
conv2d_70_69357:
conv2d_70_69359:	*
conv2d_71_69384:À@
conv2d_71_69386:@)
conv2d_72_69401:@@
conv2d_72_69403:@)
conv2d_73_69428:` 
conv2d_73_69430: )
conv2d_74_69445:  
conv2d_74_69447: )
conv2d_75_69462: 
conv2d_75_69464:
identity¢!conv2d_57/StatefulPartitionedCall¢!conv2d_58/StatefulPartitionedCall¢!conv2d_59/StatefulPartitionedCall¢!conv2d_60/StatefulPartitionedCall¢!conv2d_61/StatefulPartitionedCall¢!conv2d_62/StatefulPartitionedCall¢!conv2d_63/StatefulPartitionedCall¢!conv2d_64/StatefulPartitionedCall¢!conv2d_65/StatefulPartitionedCall¢!conv2d_66/StatefulPartitionedCall¢!conv2d_67/StatefulPartitionedCall¢!conv2d_68/StatefulPartitionedCall¢!conv2d_69/StatefulPartitionedCall¢!conv2d_70/StatefulPartitionedCall¢!conv2d_71/StatefulPartitionedCall¢!conv2d_72/StatefulPartitionedCall¢!conv2d_73/StatefulPartitionedCall¢!conv2d_74/StatefulPartitionedCall¢!conv2d_75/StatefulPartitionedCall£
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_57_69112conv2d_57_69114*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_57_layer_call_and_return_conditional_losses_691112#
!conv2d_57/StatefulPartitionedCallÇ
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0conv2d_58_69129conv2d_58_69131*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_58_layer_call_and_return_conditional_losses_691282#
!conv2d_58/StatefulPartitionedCall
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_689752"
 max_pooling2d_12/PartitionedCallÄ
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_59_69147conv2d_59_69149*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_59_layer_call_and_return_conditional_losses_691462#
!conv2d_59/StatefulPartitionedCallÅ
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0conv2d_60_69164conv2d_60_69166*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_60_layer_call_and_return_conditional_losses_691632#
!conv2d_60/StatefulPartitionedCall
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_689872"
 max_pooling2d_13/PartitionedCallÅ
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0conv2d_61_69182conv2d_61_69184*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_61_layer_call_and_return_conditional_losses_691812#
!conv2d_61/StatefulPartitionedCallÆ
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0conv2d_62_69199conv2d_62_69201*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_691982#
!conv2d_62/StatefulPartitionedCall
 max_pooling2d_14/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_689992"
 max_pooling2d_14/PartitionedCallÅ
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0conv2d_63_69217conv2d_63_69219*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_692162#
!conv2d_63/StatefulPartitionedCallÆ
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0conv2d_64_69234conv2d_64_69236*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_692332#
!conv2d_64/StatefulPartitionedCall
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_690112"
 max_pooling2d_15/PartitionedCallÅ
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_65_69252conv2d_65_69254*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_65_layer_call_and_return_conditional_losses_692512#
!conv2d_65/StatefulPartitionedCallÆ
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0conv2d_66_69269conv2d_66_69271*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_66_layer_call_and_return_conditional_losses_692682#
!conv2d_66/StatefulPartitionedCall­
 up_sampling2d_12/PartitionedCallPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_690302"
 up_sampling2d_12/PartitionedCallÁ
concatenate_12/PartitionedCallPartitionedCall)up_sampling2d_12/PartitionedCall:output:0*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_12_layer_call_and_return_conditional_losses_692822 
concatenate_12/PartitionedCallÃ
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0conv2d_67_69296conv2d_67_69298*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_67_layer_call_and_return_conditional_losses_692952#
!conv2d_67/StatefulPartitionedCallÆ
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0conv2d_68_69313conv2d_68_69315*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_68_layer_call_and_return_conditional_losses_693122#
!conv2d_68/StatefulPartitionedCall­
 up_sampling2d_13/PartitionedCallPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_690492"
 up_sampling2d_13/PartitionedCallÁ
concatenate_13/PartitionedCallPartitionedCall)up_sampling2d_13/PartitionedCall:output:0*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_13_layer_call_and_return_conditional_losses_693262 
concatenate_13/PartitionedCallÃ
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0conv2d_69_69340conv2d_69_69342*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_69_layer_call_and_return_conditional_losses_693392#
!conv2d_69/StatefulPartitionedCallÆ
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0conv2d_70_69357conv2d_70_69359*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_70_layer_call_and_return_conditional_losses_693562#
!conv2d_70/StatefulPartitionedCall­
 up_sampling2d_14/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_690682"
 up_sampling2d_14/PartitionedCallÁ
concatenate_14/PartitionedCallPartitionedCall)up_sampling2d_14/PartitionedCall:output:0*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_14_layer_call_and_return_conditional_losses_693702 
concatenate_14/PartitionedCallÂ
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0conv2d_71_69384conv2d_71_69386*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_71_layer_call_and_return_conditional_losses_693832#
!conv2d_71/StatefulPartitionedCallÅ
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0conv2d_72_69401conv2d_72_69403*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_72_layer_call_and_return_conditional_losses_694002#
!conv2d_72/StatefulPartitionedCall¬
 up_sampling2d_15/PartitionedCallPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_15_layer_call_and_return_conditional_losses_690872"
 up_sampling2d_15/PartitionedCallÂ
concatenate_15/PartitionedCallPartitionedCall)up_sampling2d_15/PartitionedCall:output:0*conv2d_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_15_layer_call_and_return_conditional_losses_694142 
concatenate_15/PartitionedCallÄ
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0conv2d_73_69428conv2d_73_69430*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_73_layer_call_and_return_conditional_losses_694272#
!conv2d_73/StatefulPartitionedCallÇ
!conv2d_74/StatefulPartitionedCallStatefulPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0conv2d_74_69445conv2d_74_69447*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_74_layer_call_and_return_conditional_losses_694442#
!conv2d_74/StatefulPartitionedCallÇ
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall*conv2d_74/StatefulPartitionedCall:output:0conv2d_75_69462conv2d_75_69464*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_75_layer_call_and_return_conditional_losses_694612#
!conv2d_75/StatefulPartitionedCall´
IdentityIdentity*conv2d_75/StatefulPartitionedCall:output:0"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall"^conv2d_74/StatefulPartitionedCall"^conv2d_75/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2F
!conv2d_74/StatefulPartitionedCall!conv2d_74/StatefulPartitionedCall2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

þ
D__inference_conv2d_71_layer_call_and_return_conditional_losses_71261

inputs9
conv2d_readvariableop_resource:À@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:À@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@À: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À
 
_user_specified_nameinputs
À
u
I__inference_concatenate_14_layer_call_and_return_conditional_losses_71241
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@@:l h
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
"
_user_specified_name
inputs/1

ÿ
D__inference_conv2d_61_layer_call_and_return_conditional_losses_71022

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @
 
_user_specified_nameinputs
Ú
L
0__inference_max_pooling2d_12_layer_call_fn_68981

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_689752
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
Z
.__inference_concatenate_14_layer_call_fn_71234
inputs_0
inputs_1
identityà
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_14_layer_call_and_return_conditional_losses_693702
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ@@@:l h
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
"
_user_specified_name
inputs/1
Í
¡
)__inference_conv2d_63_layer_call_fn_71051

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_692162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

D__inference_conv2d_68_layer_call_and_return_conditional_losses_69312

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
L
0__inference_max_pooling2d_14_layer_call_fn_69005

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_689992
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ý
D__inference_conv2d_60_layer_call_and_return_conditional_losses_69163

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
 

D__inference_conv2d_65_layer_call_and_return_conditional_losses_69251

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

)__inference_conv2d_72_layer_call_fn_71270

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_72_layer_call_and_return_conditional_losses_694002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Í
¡
)__inference_conv2d_69_layer_call_fn_71197

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_69_layer_call_and_return_conditional_losses_693392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
«
g
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_68987

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

)__inference_conv2d_59_layer_call_fn_70971

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_59_layer_call_and_return_conditional_losses_691462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ 
 
_user_specified_nameinputs
Û
©
B__inference_model_3_layer_call_and_return_conditional_losses_69959

inputs)
conv2d_57_69851: 
conv2d_57_69853: )
conv2d_58_69856:  
conv2d_58_69858: )
conv2d_59_69862: @
conv2d_59_69864:@)
conv2d_60_69867:@@
conv2d_60_69869:@*
conv2d_61_69873:@
conv2d_61_69875:	+
conv2d_62_69878:
conv2d_62_69880:	+
conv2d_63_69884:
conv2d_63_69886:	+
conv2d_64_69889:
conv2d_64_69891:	+
conv2d_65_69895:
conv2d_65_69897:	+
conv2d_66_69900:
conv2d_66_69902:	+
conv2d_67_69907:
conv2d_67_69909:	+
conv2d_68_69912:
conv2d_68_69914:	+
conv2d_69_69919:
conv2d_69_69921:	+
conv2d_70_69924:
conv2d_70_69926:	*
conv2d_71_69931:À@
conv2d_71_69933:@)
conv2d_72_69936:@@
conv2d_72_69938:@)
conv2d_73_69943:` 
conv2d_73_69945: )
conv2d_74_69948:  
conv2d_74_69950: )
conv2d_75_69953: 
conv2d_75_69955:
identity¢!conv2d_57/StatefulPartitionedCall¢!conv2d_58/StatefulPartitionedCall¢!conv2d_59/StatefulPartitionedCall¢!conv2d_60/StatefulPartitionedCall¢!conv2d_61/StatefulPartitionedCall¢!conv2d_62/StatefulPartitionedCall¢!conv2d_63/StatefulPartitionedCall¢!conv2d_64/StatefulPartitionedCall¢!conv2d_65/StatefulPartitionedCall¢!conv2d_66/StatefulPartitionedCall¢!conv2d_67/StatefulPartitionedCall¢!conv2d_68/StatefulPartitionedCall¢!conv2d_69/StatefulPartitionedCall¢!conv2d_70/StatefulPartitionedCall¢!conv2d_71/StatefulPartitionedCall¢!conv2d_72/StatefulPartitionedCall¢!conv2d_73/StatefulPartitionedCall¢!conv2d_74/StatefulPartitionedCall¢!conv2d_75/StatefulPartitionedCall£
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_57_69851conv2d_57_69853*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_57_layer_call_and_return_conditional_losses_691112#
!conv2d_57/StatefulPartitionedCallÇ
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0conv2d_58_69856conv2d_58_69858*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_58_layer_call_and_return_conditional_losses_691282#
!conv2d_58/StatefulPartitionedCall
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_689752"
 max_pooling2d_12/PartitionedCallÄ
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_59_69862conv2d_59_69864*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_59_layer_call_and_return_conditional_losses_691462#
!conv2d_59/StatefulPartitionedCallÅ
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0conv2d_60_69867conv2d_60_69869*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_60_layer_call_and_return_conditional_losses_691632#
!conv2d_60/StatefulPartitionedCall
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_689872"
 max_pooling2d_13/PartitionedCallÅ
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0conv2d_61_69873conv2d_61_69875*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_61_layer_call_and_return_conditional_losses_691812#
!conv2d_61/StatefulPartitionedCallÆ
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0conv2d_62_69878conv2d_62_69880*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_691982#
!conv2d_62/StatefulPartitionedCall
 max_pooling2d_14/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_689992"
 max_pooling2d_14/PartitionedCallÅ
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0conv2d_63_69884conv2d_63_69886*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_692162#
!conv2d_63/StatefulPartitionedCallÆ
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0conv2d_64_69889conv2d_64_69891*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_692332#
!conv2d_64/StatefulPartitionedCall
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_690112"
 max_pooling2d_15/PartitionedCallÅ
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_65_69895conv2d_65_69897*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_65_layer_call_and_return_conditional_losses_692512#
!conv2d_65/StatefulPartitionedCallÆ
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0conv2d_66_69900conv2d_66_69902*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_66_layer_call_and_return_conditional_losses_692682#
!conv2d_66/StatefulPartitionedCall­
 up_sampling2d_12/PartitionedCallPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_690302"
 up_sampling2d_12/PartitionedCallÁ
concatenate_12/PartitionedCallPartitionedCall)up_sampling2d_12/PartitionedCall:output:0*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_12_layer_call_and_return_conditional_losses_692822 
concatenate_12/PartitionedCallÃ
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0conv2d_67_69907conv2d_67_69909*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_67_layer_call_and_return_conditional_losses_692952#
!conv2d_67/StatefulPartitionedCallÆ
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0conv2d_68_69912conv2d_68_69914*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_68_layer_call_and_return_conditional_losses_693122#
!conv2d_68/StatefulPartitionedCall­
 up_sampling2d_13/PartitionedCallPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_690492"
 up_sampling2d_13/PartitionedCallÁ
concatenate_13/PartitionedCallPartitionedCall)up_sampling2d_13/PartitionedCall:output:0*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_13_layer_call_and_return_conditional_losses_693262 
concatenate_13/PartitionedCallÃ
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0conv2d_69_69919conv2d_69_69921*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_69_layer_call_and_return_conditional_losses_693392#
!conv2d_69/StatefulPartitionedCallÆ
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0conv2d_70_69924conv2d_70_69926*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_70_layer_call_and_return_conditional_losses_693562#
!conv2d_70/StatefulPartitionedCall­
 up_sampling2d_14/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_690682"
 up_sampling2d_14/PartitionedCallÁ
concatenate_14/PartitionedCallPartitionedCall)up_sampling2d_14/PartitionedCall:output:0*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_14_layer_call_and_return_conditional_losses_693702 
concatenate_14/PartitionedCallÂ
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0conv2d_71_69931conv2d_71_69933*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_71_layer_call_and_return_conditional_losses_693832#
!conv2d_71/StatefulPartitionedCallÅ
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0conv2d_72_69936conv2d_72_69938*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_72_layer_call_and_return_conditional_losses_694002#
!conv2d_72/StatefulPartitionedCall¬
 up_sampling2d_15/PartitionedCallPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_15_layer_call_and_return_conditional_losses_690872"
 up_sampling2d_15/PartitionedCallÂ
concatenate_15/PartitionedCallPartitionedCall)up_sampling2d_15/PartitionedCall:output:0*conv2d_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_15_layer_call_and_return_conditional_losses_694142 
concatenate_15/PartitionedCallÄ
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0conv2d_73_69943conv2d_73_69945*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_73_layer_call_and_return_conditional_losses_694272#
!conv2d_73/StatefulPartitionedCallÇ
!conv2d_74/StatefulPartitionedCallStatefulPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0conv2d_74_69948conv2d_74_69950*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_74_layer_call_and_return_conditional_losses_694442#
!conv2d_74/StatefulPartitionedCallÇ
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall*conv2d_74/StatefulPartitionedCall:output:0conv2d_75_69953conv2d_75_69955*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_75_layer_call_and_return_conditional_losses_694612#
!conv2d_75/StatefulPartitionedCall´
IdentityIdentity*conv2d_75/StatefulPartitionedCall:output:0"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall"^conv2d_74/StatefulPartitionedCall"^conv2d_75/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2F
!conv2d_74/StatefulPartitionedCall!conv2d_74/StatefulPartitionedCall2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

)__inference_conv2d_60_layer_call_fn_70991

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_60_layer_call_and_return_conditional_losses_691632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
Í
¡
)__inference_conv2d_68_layer_call_fn_71164

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_68_layer_call_and_return_conditional_losses_693122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
¡
)__inference_conv2d_62_layer_call_fn_71031

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_691982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Â
u
I__inference_concatenate_13_layer_call_and_return_conditional_losses_71188
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ  :l h
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/1
ª
Z
.__inference_concatenate_15_layer_call_fn_71287
inputs_0
inputs_1
identityá
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_15_layer_call_and_return_conditional_losses_694142
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ :k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1
 

D__inference_conv2d_69_layer_call_and_return_conditional_losses_69339

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
 
ý
D__inference_conv2d_73_layer_call_and_return_conditional_losses_71314

inputs8
conv2d_readvariableop_resource:` -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs

ÿ
D__inference_conv2d_61_layer_call_and_return_conditional_losses_69181

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @
 
_user_specified_nameinputs
Ö
¤"
 __inference__wrapped_model_68969
input_4J
0model_3_conv2d_57_conv2d_readvariableop_resource: ?
1model_3_conv2d_57_biasadd_readvariableop_resource: J
0model_3_conv2d_58_conv2d_readvariableop_resource:  ?
1model_3_conv2d_58_biasadd_readvariableop_resource: J
0model_3_conv2d_59_conv2d_readvariableop_resource: @?
1model_3_conv2d_59_biasadd_readvariableop_resource:@J
0model_3_conv2d_60_conv2d_readvariableop_resource:@@?
1model_3_conv2d_60_biasadd_readvariableop_resource:@K
0model_3_conv2d_61_conv2d_readvariableop_resource:@@
1model_3_conv2d_61_biasadd_readvariableop_resource:	L
0model_3_conv2d_62_conv2d_readvariableop_resource:@
1model_3_conv2d_62_biasadd_readvariableop_resource:	L
0model_3_conv2d_63_conv2d_readvariableop_resource:@
1model_3_conv2d_63_biasadd_readvariableop_resource:	L
0model_3_conv2d_64_conv2d_readvariableop_resource:@
1model_3_conv2d_64_biasadd_readvariableop_resource:	L
0model_3_conv2d_65_conv2d_readvariableop_resource:@
1model_3_conv2d_65_biasadd_readvariableop_resource:	L
0model_3_conv2d_66_conv2d_readvariableop_resource:@
1model_3_conv2d_66_biasadd_readvariableop_resource:	L
0model_3_conv2d_67_conv2d_readvariableop_resource:@
1model_3_conv2d_67_biasadd_readvariableop_resource:	L
0model_3_conv2d_68_conv2d_readvariableop_resource:@
1model_3_conv2d_68_biasadd_readvariableop_resource:	L
0model_3_conv2d_69_conv2d_readvariableop_resource:@
1model_3_conv2d_69_biasadd_readvariableop_resource:	L
0model_3_conv2d_70_conv2d_readvariableop_resource:@
1model_3_conv2d_70_biasadd_readvariableop_resource:	K
0model_3_conv2d_71_conv2d_readvariableop_resource:À@?
1model_3_conv2d_71_biasadd_readvariableop_resource:@J
0model_3_conv2d_72_conv2d_readvariableop_resource:@@?
1model_3_conv2d_72_biasadd_readvariableop_resource:@J
0model_3_conv2d_73_conv2d_readvariableop_resource:` ?
1model_3_conv2d_73_biasadd_readvariableop_resource: J
0model_3_conv2d_74_conv2d_readvariableop_resource:  ?
1model_3_conv2d_74_biasadd_readvariableop_resource: J
0model_3_conv2d_75_conv2d_readvariableop_resource: ?
1model_3_conv2d_75_biasadd_readvariableop_resource:
identity¢(model_3/conv2d_57/BiasAdd/ReadVariableOp¢'model_3/conv2d_57/Conv2D/ReadVariableOp¢(model_3/conv2d_58/BiasAdd/ReadVariableOp¢'model_3/conv2d_58/Conv2D/ReadVariableOp¢(model_3/conv2d_59/BiasAdd/ReadVariableOp¢'model_3/conv2d_59/Conv2D/ReadVariableOp¢(model_3/conv2d_60/BiasAdd/ReadVariableOp¢'model_3/conv2d_60/Conv2D/ReadVariableOp¢(model_3/conv2d_61/BiasAdd/ReadVariableOp¢'model_3/conv2d_61/Conv2D/ReadVariableOp¢(model_3/conv2d_62/BiasAdd/ReadVariableOp¢'model_3/conv2d_62/Conv2D/ReadVariableOp¢(model_3/conv2d_63/BiasAdd/ReadVariableOp¢'model_3/conv2d_63/Conv2D/ReadVariableOp¢(model_3/conv2d_64/BiasAdd/ReadVariableOp¢'model_3/conv2d_64/Conv2D/ReadVariableOp¢(model_3/conv2d_65/BiasAdd/ReadVariableOp¢'model_3/conv2d_65/Conv2D/ReadVariableOp¢(model_3/conv2d_66/BiasAdd/ReadVariableOp¢'model_3/conv2d_66/Conv2D/ReadVariableOp¢(model_3/conv2d_67/BiasAdd/ReadVariableOp¢'model_3/conv2d_67/Conv2D/ReadVariableOp¢(model_3/conv2d_68/BiasAdd/ReadVariableOp¢'model_3/conv2d_68/Conv2D/ReadVariableOp¢(model_3/conv2d_69/BiasAdd/ReadVariableOp¢'model_3/conv2d_69/Conv2D/ReadVariableOp¢(model_3/conv2d_70/BiasAdd/ReadVariableOp¢'model_3/conv2d_70/Conv2D/ReadVariableOp¢(model_3/conv2d_71/BiasAdd/ReadVariableOp¢'model_3/conv2d_71/Conv2D/ReadVariableOp¢(model_3/conv2d_72/BiasAdd/ReadVariableOp¢'model_3/conv2d_72/Conv2D/ReadVariableOp¢(model_3/conv2d_73/BiasAdd/ReadVariableOp¢'model_3/conv2d_73/Conv2D/ReadVariableOp¢(model_3/conv2d_74/BiasAdd/ReadVariableOp¢'model_3/conv2d_74/Conv2D/ReadVariableOp¢(model_3/conv2d_75/BiasAdd/ReadVariableOp¢'model_3/conv2d_75/Conv2D/ReadVariableOpË
'model_3/conv2d_57/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_57_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_3/conv2d_57/Conv2D/ReadVariableOpÜ
model_3/conv2d_57/Conv2DConv2Dinput_4/model_3/conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
model_3/conv2d_57/Conv2DÂ
(model_3/conv2d_57/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_57_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_57/BiasAdd/ReadVariableOpÒ
model_3/conv2d_57/BiasAddBiasAdd!model_3/conv2d_57/Conv2D:output:00model_3/conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_3/conv2d_57/BiasAdd
model_3/conv2d_57/ReluRelu"model_3/conv2d_57/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_3/conv2d_57/ReluË
'model_3/conv2d_58/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_3/conv2d_58/Conv2D/ReadVariableOpù
model_3/conv2d_58/Conv2DConv2D$model_3/conv2d_57/Relu:activations:0/model_3/conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
model_3/conv2d_58/Conv2DÂ
(model_3/conv2d_58/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_58/BiasAdd/ReadVariableOpÒ
model_3/conv2d_58/BiasAddBiasAdd!model_3/conv2d_58/Conv2D:output:00model_3/conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_3/conv2d_58/BiasAdd
model_3/conv2d_58/ReluRelu"model_3/conv2d_58/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_3/conv2d_58/Reluâ
 model_3/max_pooling2d_12/MaxPoolMaxPool$model_3/conv2d_58/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *
ksize
*
paddingVALID*
strides
2"
 model_3/max_pooling2d_12/MaxPoolË
'model_3/conv2d_59/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'model_3/conv2d_59/Conv2D/ReadVariableOpü
model_3/conv2d_59/Conv2DConv2D)model_3/max_pooling2d_12/MaxPool:output:0/model_3/conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
model_3/conv2d_59/Conv2DÂ
(model_3/conv2d_59/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_59/BiasAdd/ReadVariableOpÐ
model_3/conv2d_59/BiasAddBiasAdd!model_3/conv2d_59/Conv2D:output:00model_3/conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
model_3/conv2d_59/BiasAdd
model_3/conv2d_59/ReluRelu"model_3/conv2d_59/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
model_3/conv2d_59/ReluË
'model_3/conv2d_60/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_3/conv2d_60/Conv2D/ReadVariableOp÷
model_3/conv2d_60/Conv2DConv2D$model_3/conv2d_59/Relu:activations:0/model_3/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
model_3/conv2d_60/Conv2DÂ
(model_3/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_60/BiasAdd/ReadVariableOpÐ
model_3/conv2d_60/BiasAddBiasAdd!model_3/conv2d_60/Conv2D:output:00model_3/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
model_3/conv2d_60/BiasAdd
model_3/conv2d_60/ReluRelu"model_3/conv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
model_3/conv2d_60/Reluâ
 model_3/max_pooling2d_13/MaxPoolMaxPool$model_3/conv2d_60/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
ksize
*
paddingVALID*
strides
2"
 model_3/max_pooling2d_13/MaxPoolÌ
'model_3/conv2d_61/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'model_3/conv2d_61/Conv2D/ReadVariableOpý
model_3/conv2d_61/Conv2DConv2D)model_3/max_pooling2d_13/MaxPool:output:0/model_3/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
model_3/conv2d_61/Conv2DÃ
(model_3/conv2d_61/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_61_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_61/BiasAdd/ReadVariableOpÑ
model_3/conv2d_61/BiasAddBiasAdd!model_3/conv2d_61/Conv2D:output:00model_3/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
model_3/conv2d_61/BiasAdd
model_3/conv2d_61/ReluRelu"model_3/conv2d_61/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
model_3/conv2d_61/ReluÍ
'model_3/conv2d_62/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_62/Conv2D/ReadVariableOpø
model_3/conv2d_62/Conv2DConv2D$model_3/conv2d_61/Relu:activations:0/model_3/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
model_3/conv2d_62/Conv2DÃ
(model_3/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_62/BiasAdd/ReadVariableOpÑ
model_3/conv2d_62/BiasAddBiasAdd!model_3/conv2d_62/Conv2D:output:00model_3/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
model_3/conv2d_62/BiasAdd
model_3/conv2d_62/ReluRelu"model_3/conv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
model_3/conv2d_62/Reluã
 model_3/max_pooling2d_14/MaxPoolMaxPool$model_3/conv2d_62/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2"
 model_3/max_pooling2d_14/MaxPoolÍ
'model_3/conv2d_63/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_63/Conv2D/ReadVariableOpý
model_3/conv2d_63/Conv2DConv2D)model_3/max_pooling2d_14/MaxPool:output:0/model_3/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
model_3/conv2d_63/Conv2DÃ
(model_3/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_63/BiasAdd/ReadVariableOpÑ
model_3/conv2d_63/BiasAddBiasAdd!model_3/conv2d_63/Conv2D:output:00model_3/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_63/BiasAdd
model_3/conv2d_63/ReluRelu"model_3/conv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_63/ReluÍ
'model_3/conv2d_64/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_64_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_64/Conv2D/ReadVariableOpø
model_3/conv2d_64/Conv2DConv2D$model_3/conv2d_63/Relu:activations:0/model_3/conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
model_3/conv2d_64/Conv2DÃ
(model_3/conv2d_64/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_64_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_64/BiasAdd/ReadVariableOpÑ
model_3/conv2d_64/BiasAddBiasAdd!model_3/conv2d_64/Conv2D:output:00model_3/conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_64/BiasAdd
model_3/conv2d_64/ReluRelu"model_3/conv2d_64/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_64/Reluã
 model_3/max_pooling2d_15/MaxPoolMaxPool$model_3/conv2d_64/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2"
 model_3/max_pooling2d_15/MaxPoolÍ
'model_3/conv2d_65/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_65_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_65/Conv2D/ReadVariableOpý
model_3/conv2d_65/Conv2DConv2D)model_3/max_pooling2d_15/MaxPool:output:0/model_3/conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
model_3/conv2d_65/Conv2DÃ
(model_3/conv2d_65/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_65_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_65/BiasAdd/ReadVariableOpÑ
model_3/conv2d_65/BiasAddBiasAdd!model_3/conv2d_65/Conv2D:output:00model_3/conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_65/BiasAdd
model_3/conv2d_65/ReluRelu"model_3/conv2d_65/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_65/ReluÍ
'model_3/conv2d_66/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_66_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_66/Conv2D/ReadVariableOpø
model_3/conv2d_66/Conv2DConv2D$model_3/conv2d_65/Relu:activations:0/model_3/conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
model_3/conv2d_66/Conv2DÃ
(model_3/conv2d_66/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_66_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_66/BiasAdd/ReadVariableOpÑ
model_3/conv2d_66/BiasAddBiasAdd!model_3/conv2d_66/Conv2D:output:00model_3/conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_66/BiasAdd
model_3/conv2d_66/ReluRelu"model_3/conv2d_66/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_66/Relu
model_3/up_sampling2d_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2 
model_3/up_sampling2d_12/Const
 model_3/up_sampling2d_12/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2"
 model_3/up_sampling2d_12/Const_1¼
model_3/up_sampling2d_12/mulMul'model_3/up_sampling2d_12/Const:output:0)model_3/up_sampling2d_12/Const_1:output:0*
T0*
_output_shapes
:2
model_3/up_sampling2d_12/mul¤
5model_3/up_sampling2d_12/resize/ResizeNearestNeighborResizeNearestNeighbor$model_3/conv2d_66/Relu:activations:0 model_3/up_sampling2d_12/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(27
5model_3/up_sampling2d_12/resize/ResizeNearestNeighbor
"model_3/concatenate_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_3/concatenate_12/concat/axis©
model_3/concatenate_12/concatConcatV2Fmodel_3/up_sampling2d_12/resize/ResizeNearestNeighbor:resized_images:0$model_3/conv2d_64/Relu:activations:0+model_3/concatenate_12/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/concatenate_12/concatÍ
'model_3/conv2d_67/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_67_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_67/Conv2D/ReadVariableOpú
model_3/conv2d_67/Conv2DConv2D&model_3/concatenate_12/concat:output:0/model_3/conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
model_3/conv2d_67/Conv2DÃ
(model_3/conv2d_67/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_67_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_67/BiasAdd/ReadVariableOpÑ
model_3/conv2d_67/BiasAddBiasAdd!model_3/conv2d_67/Conv2D:output:00model_3/conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_67/BiasAdd
model_3/conv2d_67/ReluRelu"model_3/conv2d_67/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_67/ReluÍ
'model_3/conv2d_68/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_68_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_68/Conv2D/ReadVariableOpø
model_3/conv2d_68/Conv2DConv2D$model_3/conv2d_67/Relu:activations:0/model_3/conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
model_3/conv2d_68/Conv2DÃ
(model_3/conv2d_68/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_68_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_68/BiasAdd/ReadVariableOpÑ
model_3/conv2d_68/BiasAddBiasAdd!model_3/conv2d_68/Conv2D:output:00model_3/conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_68/BiasAdd
model_3/conv2d_68/ReluRelu"model_3/conv2d_68/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_68/Relu
model_3/up_sampling2d_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2 
model_3/up_sampling2d_13/Const
 model_3/up_sampling2d_13/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2"
 model_3/up_sampling2d_13/Const_1¼
model_3/up_sampling2d_13/mulMul'model_3/up_sampling2d_13/Const:output:0)model_3/up_sampling2d_13/Const_1:output:0*
T0*
_output_shapes
:2
model_3/up_sampling2d_13/mul¤
5model_3/up_sampling2d_13/resize/ResizeNearestNeighborResizeNearestNeighbor$model_3/conv2d_68/Relu:activations:0 model_3/up_sampling2d_13/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(27
5model_3/up_sampling2d_13/resize/ResizeNearestNeighbor
"model_3/concatenate_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_3/concatenate_13/concat/axis©
model_3/concatenate_13/concatConcatV2Fmodel_3/up_sampling2d_13/resize/ResizeNearestNeighbor:resized_images:0$model_3/conv2d_62/Relu:activations:0+model_3/concatenate_13/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
model_3/concatenate_13/concatÍ
'model_3/conv2d_69/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_69_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_69/Conv2D/ReadVariableOpú
model_3/conv2d_69/Conv2DConv2D&model_3/concatenate_13/concat:output:0/model_3/conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
model_3/conv2d_69/Conv2DÃ
(model_3/conv2d_69/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_69_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_69/BiasAdd/ReadVariableOpÑ
model_3/conv2d_69/BiasAddBiasAdd!model_3/conv2d_69/Conv2D:output:00model_3/conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
model_3/conv2d_69/BiasAdd
model_3/conv2d_69/ReluRelu"model_3/conv2d_69/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
model_3/conv2d_69/ReluÍ
'model_3/conv2d_70/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_70_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_70/Conv2D/ReadVariableOpø
model_3/conv2d_70/Conv2DConv2D$model_3/conv2d_69/Relu:activations:0/model_3/conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
model_3/conv2d_70/Conv2DÃ
(model_3/conv2d_70/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_70_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_70/BiasAdd/ReadVariableOpÑ
model_3/conv2d_70/BiasAddBiasAdd!model_3/conv2d_70/Conv2D:output:00model_3/conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
model_3/conv2d_70/BiasAdd
model_3/conv2d_70/ReluRelu"model_3/conv2d_70/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
model_3/conv2d_70/Relu
model_3/up_sampling2d_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2 
model_3/up_sampling2d_14/Const
 model_3/up_sampling2d_14/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2"
 model_3/up_sampling2d_14/Const_1¼
model_3/up_sampling2d_14/mulMul'model_3/up_sampling2d_14/Const:output:0)model_3/up_sampling2d_14/Const_1:output:0*
T0*
_output_shapes
:2
model_3/up_sampling2d_14/mul¤
5model_3/up_sampling2d_14/resize/ResizeNearestNeighborResizeNearestNeighbor$model_3/conv2d_70/Relu:activations:0 model_3/up_sampling2d_14/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
half_pixel_centers(27
5model_3/up_sampling2d_14/resize/ResizeNearestNeighbor
"model_3/concatenate_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_3/concatenate_14/concat/axis©
model_3/concatenate_14/concatConcatV2Fmodel_3/up_sampling2d_14/resize/ResizeNearestNeighbor:resized_images:0$model_3/conv2d_60/Relu:activations:0+model_3/concatenate_14/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À2
model_3/concatenate_14/concatÌ
'model_3/conv2d_71/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_71_conv2d_readvariableop_resource*'
_output_shapes
:À@*
dtype02)
'model_3/conv2d_71/Conv2D/ReadVariableOpù
model_3/conv2d_71/Conv2DConv2D&model_3/concatenate_14/concat:output:0/model_3/conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
model_3/conv2d_71/Conv2DÂ
(model_3/conv2d_71/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_71_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_71/BiasAdd/ReadVariableOpÐ
model_3/conv2d_71/BiasAddBiasAdd!model_3/conv2d_71/Conv2D:output:00model_3/conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
model_3/conv2d_71/BiasAdd
model_3/conv2d_71/ReluRelu"model_3/conv2d_71/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
model_3/conv2d_71/ReluË
'model_3/conv2d_72/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_3/conv2d_72/Conv2D/ReadVariableOp÷
model_3/conv2d_72/Conv2DConv2D$model_3/conv2d_71/Relu:activations:0/model_3/conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
model_3/conv2d_72/Conv2DÂ
(model_3/conv2d_72/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_72_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_72/BiasAdd/ReadVariableOpÐ
model_3/conv2d_72/BiasAddBiasAdd!model_3/conv2d_72/Conv2D:output:00model_3/conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
model_3/conv2d_72/BiasAdd
model_3/conv2d_72/ReluRelu"model_3/conv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
model_3/conv2d_72/Relu
model_3/up_sampling2d_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   2 
model_3/up_sampling2d_15/Const
 model_3/up_sampling2d_15/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2"
 model_3/up_sampling2d_15/Const_1¼
model_3/up_sampling2d_15/mulMul'model_3/up_sampling2d_15/Const:output:0)model_3/up_sampling2d_15/Const_1:output:0*
T0*
_output_shapes
:2
model_3/up_sampling2d_15/mul¥
5model_3/up_sampling2d_15/resize/ResizeNearestNeighborResizeNearestNeighbor$model_3/conv2d_72/Relu:activations:0 model_3/up_sampling2d_15/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(27
5model_3/up_sampling2d_15/resize/ResizeNearestNeighbor
"model_3/concatenate_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_3/concatenate_15/concat/axisª
model_3/concatenate_15/concatConcatV2Fmodel_3/up_sampling2d_15/resize/ResizeNearestNeighbor:resized_images:0$model_3/conv2d_58/Relu:activations:0+model_3/concatenate_15/concat/axis:output:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
model_3/concatenate_15/concatË
'model_3/conv2d_73/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02)
'model_3/conv2d_73/Conv2D/ReadVariableOpû
model_3/conv2d_73/Conv2DConv2D&model_3/concatenate_15/concat:output:0/model_3/conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
model_3/conv2d_73/Conv2DÂ
(model_3/conv2d_73/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_73/BiasAdd/ReadVariableOpÒ
model_3/conv2d_73/BiasAddBiasAdd!model_3/conv2d_73/Conv2D:output:00model_3/conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_3/conv2d_73/BiasAdd
model_3/conv2d_73/ReluRelu"model_3/conv2d_73/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_3/conv2d_73/ReluË
'model_3/conv2d_74/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_3/conv2d_74/Conv2D/ReadVariableOpù
model_3/conv2d_74/Conv2DConv2D$model_3/conv2d_73/Relu:activations:0/model_3/conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
model_3/conv2d_74/Conv2DÂ
(model_3/conv2d_74/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_3/conv2d_74/BiasAdd/ReadVariableOpÒ
model_3/conv2d_74/BiasAddBiasAdd!model_3/conv2d_74/Conv2D:output:00model_3/conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_3/conv2d_74/BiasAdd
model_3/conv2d_74/ReluRelu"model_3/conv2d_74/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_3/conv2d_74/ReluË
'model_3/conv2d_75/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_75_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_3/conv2d_75/Conv2D/ReadVariableOpù
model_3/conv2d_75/Conv2DConv2D$model_3/conv2d_74/Relu:activations:0/model_3/conv2d_75/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
model_3/conv2d_75/Conv2DÂ
(model_3/conv2d_75/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_75_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_3/conv2d_75/BiasAdd/ReadVariableOpÒ
model_3/conv2d_75/BiasAddBiasAdd!model_3/conv2d_75/Conv2D:output:00model_3/conv2d_75/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_75/BiasAdd¡
model_3/conv2d_75/SigmoidSigmoid"model_3/conv2d_75/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_3/conv2d_75/SigmoidÊ
IdentityIdentitymodel_3/conv2d_75/Sigmoid:y:0)^model_3/conv2d_57/BiasAdd/ReadVariableOp(^model_3/conv2d_57/Conv2D/ReadVariableOp)^model_3/conv2d_58/BiasAdd/ReadVariableOp(^model_3/conv2d_58/Conv2D/ReadVariableOp)^model_3/conv2d_59/BiasAdd/ReadVariableOp(^model_3/conv2d_59/Conv2D/ReadVariableOp)^model_3/conv2d_60/BiasAdd/ReadVariableOp(^model_3/conv2d_60/Conv2D/ReadVariableOp)^model_3/conv2d_61/BiasAdd/ReadVariableOp(^model_3/conv2d_61/Conv2D/ReadVariableOp)^model_3/conv2d_62/BiasAdd/ReadVariableOp(^model_3/conv2d_62/Conv2D/ReadVariableOp)^model_3/conv2d_63/BiasAdd/ReadVariableOp(^model_3/conv2d_63/Conv2D/ReadVariableOp)^model_3/conv2d_64/BiasAdd/ReadVariableOp(^model_3/conv2d_64/Conv2D/ReadVariableOp)^model_3/conv2d_65/BiasAdd/ReadVariableOp(^model_3/conv2d_65/Conv2D/ReadVariableOp)^model_3/conv2d_66/BiasAdd/ReadVariableOp(^model_3/conv2d_66/Conv2D/ReadVariableOp)^model_3/conv2d_67/BiasAdd/ReadVariableOp(^model_3/conv2d_67/Conv2D/ReadVariableOp)^model_3/conv2d_68/BiasAdd/ReadVariableOp(^model_3/conv2d_68/Conv2D/ReadVariableOp)^model_3/conv2d_69/BiasAdd/ReadVariableOp(^model_3/conv2d_69/Conv2D/ReadVariableOp)^model_3/conv2d_70/BiasAdd/ReadVariableOp(^model_3/conv2d_70/Conv2D/ReadVariableOp)^model_3/conv2d_71/BiasAdd/ReadVariableOp(^model_3/conv2d_71/Conv2D/ReadVariableOp)^model_3/conv2d_72/BiasAdd/ReadVariableOp(^model_3/conv2d_72/Conv2D/ReadVariableOp)^model_3/conv2d_73/BiasAdd/ReadVariableOp(^model_3/conv2d_73/Conv2D/ReadVariableOp)^model_3/conv2d_74/BiasAdd/ReadVariableOp(^model_3/conv2d_74/Conv2D/ReadVariableOp)^model_3/conv2d_75/BiasAdd/ReadVariableOp(^model_3/conv2d_75/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(model_3/conv2d_57/BiasAdd/ReadVariableOp(model_3/conv2d_57/BiasAdd/ReadVariableOp2R
'model_3/conv2d_57/Conv2D/ReadVariableOp'model_3/conv2d_57/Conv2D/ReadVariableOp2T
(model_3/conv2d_58/BiasAdd/ReadVariableOp(model_3/conv2d_58/BiasAdd/ReadVariableOp2R
'model_3/conv2d_58/Conv2D/ReadVariableOp'model_3/conv2d_58/Conv2D/ReadVariableOp2T
(model_3/conv2d_59/BiasAdd/ReadVariableOp(model_3/conv2d_59/BiasAdd/ReadVariableOp2R
'model_3/conv2d_59/Conv2D/ReadVariableOp'model_3/conv2d_59/Conv2D/ReadVariableOp2T
(model_3/conv2d_60/BiasAdd/ReadVariableOp(model_3/conv2d_60/BiasAdd/ReadVariableOp2R
'model_3/conv2d_60/Conv2D/ReadVariableOp'model_3/conv2d_60/Conv2D/ReadVariableOp2T
(model_3/conv2d_61/BiasAdd/ReadVariableOp(model_3/conv2d_61/BiasAdd/ReadVariableOp2R
'model_3/conv2d_61/Conv2D/ReadVariableOp'model_3/conv2d_61/Conv2D/ReadVariableOp2T
(model_3/conv2d_62/BiasAdd/ReadVariableOp(model_3/conv2d_62/BiasAdd/ReadVariableOp2R
'model_3/conv2d_62/Conv2D/ReadVariableOp'model_3/conv2d_62/Conv2D/ReadVariableOp2T
(model_3/conv2d_63/BiasAdd/ReadVariableOp(model_3/conv2d_63/BiasAdd/ReadVariableOp2R
'model_3/conv2d_63/Conv2D/ReadVariableOp'model_3/conv2d_63/Conv2D/ReadVariableOp2T
(model_3/conv2d_64/BiasAdd/ReadVariableOp(model_3/conv2d_64/BiasAdd/ReadVariableOp2R
'model_3/conv2d_64/Conv2D/ReadVariableOp'model_3/conv2d_64/Conv2D/ReadVariableOp2T
(model_3/conv2d_65/BiasAdd/ReadVariableOp(model_3/conv2d_65/BiasAdd/ReadVariableOp2R
'model_3/conv2d_65/Conv2D/ReadVariableOp'model_3/conv2d_65/Conv2D/ReadVariableOp2T
(model_3/conv2d_66/BiasAdd/ReadVariableOp(model_3/conv2d_66/BiasAdd/ReadVariableOp2R
'model_3/conv2d_66/Conv2D/ReadVariableOp'model_3/conv2d_66/Conv2D/ReadVariableOp2T
(model_3/conv2d_67/BiasAdd/ReadVariableOp(model_3/conv2d_67/BiasAdd/ReadVariableOp2R
'model_3/conv2d_67/Conv2D/ReadVariableOp'model_3/conv2d_67/Conv2D/ReadVariableOp2T
(model_3/conv2d_68/BiasAdd/ReadVariableOp(model_3/conv2d_68/BiasAdd/ReadVariableOp2R
'model_3/conv2d_68/Conv2D/ReadVariableOp'model_3/conv2d_68/Conv2D/ReadVariableOp2T
(model_3/conv2d_69/BiasAdd/ReadVariableOp(model_3/conv2d_69/BiasAdd/ReadVariableOp2R
'model_3/conv2d_69/Conv2D/ReadVariableOp'model_3/conv2d_69/Conv2D/ReadVariableOp2T
(model_3/conv2d_70/BiasAdd/ReadVariableOp(model_3/conv2d_70/BiasAdd/ReadVariableOp2R
'model_3/conv2d_70/Conv2D/ReadVariableOp'model_3/conv2d_70/Conv2D/ReadVariableOp2T
(model_3/conv2d_71/BiasAdd/ReadVariableOp(model_3/conv2d_71/BiasAdd/ReadVariableOp2R
'model_3/conv2d_71/Conv2D/ReadVariableOp'model_3/conv2d_71/Conv2D/ReadVariableOp2T
(model_3/conv2d_72/BiasAdd/ReadVariableOp(model_3/conv2d_72/BiasAdd/ReadVariableOp2R
'model_3/conv2d_72/Conv2D/ReadVariableOp'model_3/conv2d_72/Conv2D/ReadVariableOp2T
(model_3/conv2d_73/BiasAdd/ReadVariableOp(model_3/conv2d_73/BiasAdd/ReadVariableOp2R
'model_3/conv2d_73/Conv2D/ReadVariableOp'model_3/conv2d_73/Conv2D/ReadVariableOp2T
(model_3/conv2d_74/BiasAdd/ReadVariableOp(model_3/conv2d_74/BiasAdd/ReadVariableOp2R
'model_3/conv2d_74/Conv2D/ReadVariableOp'model_3/conv2d_74/Conv2D/ReadVariableOp2T
(model_3/conv2d_75/BiasAdd/ReadVariableOp(model_3/conv2d_75/BiasAdd/ReadVariableOp2R
'model_3/conv2d_75/Conv2D/ReadVariableOp'model_3/conv2d_75/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
Î

)__inference_conv2d_73_layer_call_fn_71303

inputs!
unknown:` 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_73_layer_call_and_return_conditional_losses_694272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ`: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
ó
®

#__inference_signature_wrapper_70430
input_4!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	%

unknown_27:À@

unknown_28:@$

unknown_29:@@

unknown_30:@$

unknown_31:` 

unknown_32: $

unknown_33:  

unknown_34: $

unknown_35: 

unknown_36:
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_689692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
Þ
ª
B__inference_model_3_layer_call_and_return_conditional_losses_70230
input_4)
conv2d_57_70122: 
conv2d_57_70124: )
conv2d_58_70127:  
conv2d_58_70129: )
conv2d_59_70133: @
conv2d_59_70135:@)
conv2d_60_70138:@@
conv2d_60_70140:@*
conv2d_61_70144:@
conv2d_61_70146:	+
conv2d_62_70149:
conv2d_62_70151:	+
conv2d_63_70155:
conv2d_63_70157:	+
conv2d_64_70160:
conv2d_64_70162:	+
conv2d_65_70166:
conv2d_65_70168:	+
conv2d_66_70171:
conv2d_66_70173:	+
conv2d_67_70178:
conv2d_67_70180:	+
conv2d_68_70183:
conv2d_68_70185:	+
conv2d_69_70190:
conv2d_69_70192:	+
conv2d_70_70195:
conv2d_70_70197:	*
conv2d_71_70202:À@
conv2d_71_70204:@)
conv2d_72_70207:@@
conv2d_72_70209:@)
conv2d_73_70214:` 
conv2d_73_70216: )
conv2d_74_70219:  
conv2d_74_70221: )
conv2d_75_70224: 
conv2d_75_70226:
identity¢!conv2d_57/StatefulPartitionedCall¢!conv2d_58/StatefulPartitionedCall¢!conv2d_59/StatefulPartitionedCall¢!conv2d_60/StatefulPartitionedCall¢!conv2d_61/StatefulPartitionedCall¢!conv2d_62/StatefulPartitionedCall¢!conv2d_63/StatefulPartitionedCall¢!conv2d_64/StatefulPartitionedCall¢!conv2d_65/StatefulPartitionedCall¢!conv2d_66/StatefulPartitionedCall¢!conv2d_67/StatefulPartitionedCall¢!conv2d_68/StatefulPartitionedCall¢!conv2d_69/StatefulPartitionedCall¢!conv2d_70/StatefulPartitionedCall¢!conv2d_71/StatefulPartitionedCall¢!conv2d_72/StatefulPartitionedCall¢!conv2d_73/StatefulPartitionedCall¢!conv2d_74/StatefulPartitionedCall¢!conv2d_75/StatefulPartitionedCall¤
!conv2d_57/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_57_70122conv2d_57_70124*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_57_layer_call_and_return_conditional_losses_691112#
!conv2d_57/StatefulPartitionedCallÇ
!conv2d_58/StatefulPartitionedCallStatefulPartitionedCall*conv2d_57/StatefulPartitionedCall:output:0conv2d_58_70127conv2d_58_70129*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_58_layer_call_and_return_conditional_losses_691282#
!conv2d_58/StatefulPartitionedCall
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_689752"
 max_pooling2d_12/PartitionedCallÄ
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_59_70133conv2d_59_70135*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_59_layer_call_and_return_conditional_losses_691462#
!conv2d_59/StatefulPartitionedCallÅ
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0conv2d_60_70138conv2d_60_70140*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_60_layer_call_and_return_conditional_losses_691632#
!conv2d_60/StatefulPartitionedCall
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_689872"
 max_pooling2d_13/PartitionedCallÅ
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0conv2d_61_70144conv2d_61_70146*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_61_layer_call_and_return_conditional_losses_691812#
!conv2d_61/StatefulPartitionedCallÆ
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0conv2d_62_70149conv2d_62_70151*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_691982#
!conv2d_62/StatefulPartitionedCall
 max_pooling2d_14/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_689992"
 max_pooling2d_14/PartitionedCallÅ
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0conv2d_63_70155conv2d_63_70157*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_692162#
!conv2d_63/StatefulPartitionedCallÆ
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0conv2d_64_70160conv2d_64_70162*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_692332#
!conv2d_64/StatefulPartitionedCall
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_690112"
 max_pooling2d_15/PartitionedCallÅ
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_65_70166conv2d_65_70168*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_65_layer_call_and_return_conditional_losses_692512#
!conv2d_65/StatefulPartitionedCallÆ
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0conv2d_66_70171conv2d_66_70173*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_66_layer_call_and_return_conditional_losses_692682#
!conv2d_66/StatefulPartitionedCall­
 up_sampling2d_12/PartitionedCallPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_690302"
 up_sampling2d_12/PartitionedCallÁ
concatenate_12/PartitionedCallPartitionedCall)up_sampling2d_12/PartitionedCall:output:0*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_12_layer_call_and_return_conditional_losses_692822 
concatenate_12/PartitionedCallÃ
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall'concatenate_12/PartitionedCall:output:0conv2d_67_70178conv2d_67_70180*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_67_layer_call_and_return_conditional_losses_692952#
!conv2d_67/StatefulPartitionedCallÆ
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0conv2d_68_70183conv2d_68_70185*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_68_layer_call_and_return_conditional_losses_693122#
!conv2d_68/StatefulPartitionedCall­
 up_sampling2d_13/PartitionedCallPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_690492"
 up_sampling2d_13/PartitionedCallÁ
concatenate_13/PartitionedCallPartitionedCall)up_sampling2d_13/PartitionedCall:output:0*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_13_layer_call_and_return_conditional_losses_693262 
concatenate_13/PartitionedCallÃ
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall'concatenate_13/PartitionedCall:output:0conv2d_69_70190conv2d_69_70192*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_69_layer_call_and_return_conditional_losses_693392#
!conv2d_69/StatefulPartitionedCallÆ
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0conv2d_70_70195conv2d_70_70197*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_70_layer_call_and_return_conditional_losses_693562#
!conv2d_70/StatefulPartitionedCall­
 up_sampling2d_14/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_690682"
 up_sampling2d_14/PartitionedCallÁ
concatenate_14/PartitionedCallPartitionedCall)up_sampling2d_14/PartitionedCall:output:0*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_14_layer_call_and_return_conditional_losses_693702 
concatenate_14/PartitionedCallÂ
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall'concatenate_14/PartitionedCall:output:0conv2d_71_70202conv2d_71_70204*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_71_layer_call_and_return_conditional_losses_693832#
!conv2d_71/StatefulPartitionedCallÅ
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0conv2d_72_70207conv2d_72_70209*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_72_layer_call_and_return_conditional_losses_694002#
!conv2d_72/StatefulPartitionedCall¬
 up_sampling2d_15/PartitionedCallPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_up_sampling2d_15_layer_call_and_return_conditional_losses_690872"
 up_sampling2d_15/PartitionedCallÂ
concatenate_15/PartitionedCallPartitionedCall)up_sampling2d_15/PartitionedCall:output:0*conv2d_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_15_layer_call_and_return_conditional_losses_694142 
concatenate_15/PartitionedCallÄ
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0conv2d_73_70214conv2d_73_70216*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_73_layer_call_and_return_conditional_losses_694272#
!conv2d_73/StatefulPartitionedCallÇ
!conv2d_74/StatefulPartitionedCallStatefulPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0conv2d_74_70219conv2d_74_70221*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_74_layer_call_and_return_conditional_losses_694442#
!conv2d_74/StatefulPartitionedCallÇ
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall*conv2d_74/StatefulPartitionedCall:output:0conv2d_75_70224conv2d_75_70226*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_75_layer_call_and_return_conditional_losses_694612#
!conv2d_75/StatefulPartitionedCall´
IdentityIdentity*conv2d_75/StatefulPartitionedCall:output:0"^conv2d_57/StatefulPartitionedCall"^conv2d_58/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall"^conv2d_74/StatefulPartitionedCall"^conv2d_75/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_57/StatefulPartitionedCall!conv2d_57/StatefulPartitionedCall2F
!conv2d_58/StatefulPartitionedCall!conv2d_58/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2F
!conv2d_74/StatefulPartitionedCall!conv2d_74/StatefulPartitionedCall2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
 

D__inference_conv2d_67_layer_call_and_return_conditional_losses_71155

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

)__inference_conv2d_75_layer_call_fn_71343

inputs!
unknown: 
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_75_layer_call_and_return_conditional_losses_694612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
²
g
K__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_69068

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
g
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_68975

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

²

'__inference_model_3_layer_call_fn_70119
input_4!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@$
	unknown_7:@
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	%

unknown_27:À@

unknown_28:@$

unknown_29:@@

unknown_30:@$

unknown_31:` 

unknown_32: $

unknown_33:  

unknown_34: $

unknown_35: 

unknown_36:
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_699592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
²
g
K__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_69030

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
g
K__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_69049

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Î
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

D__inference_conv2d_70_layer_call_and_return_conditional_losses_71228

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
É

)__inference_conv2d_71_layer_call_fn_71250

inputs"
unknown:À@
	unknown_0:@
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_71_layer_call_and_return_conditional_losses_693832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ@@À: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À
 
_user_specified_nameinputs
 
ý
D__inference_conv2d_57_layer_call_and_return_conditional_losses_69111

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ý
D__inference_conv2d_72_layer_call_and_return_conditional_losses_71281

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs
ë
å
B__inference_model_3_layer_call_and_return_conditional_losses_70757

inputsB
(conv2d_57_conv2d_readvariableop_resource: 7
)conv2d_57_biasadd_readvariableop_resource: B
(conv2d_58_conv2d_readvariableop_resource:  7
)conv2d_58_biasadd_readvariableop_resource: B
(conv2d_59_conv2d_readvariableop_resource: @7
)conv2d_59_biasadd_readvariableop_resource:@B
(conv2d_60_conv2d_readvariableop_resource:@@7
)conv2d_60_biasadd_readvariableop_resource:@C
(conv2d_61_conv2d_readvariableop_resource:@8
)conv2d_61_biasadd_readvariableop_resource:	D
(conv2d_62_conv2d_readvariableop_resource:8
)conv2d_62_biasadd_readvariableop_resource:	D
(conv2d_63_conv2d_readvariableop_resource:8
)conv2d_63_biasadd_readvariableop_resource:	D
(conv2d_64_conv2d_readvariableop_resource:8
)conv2d_64_biasadd_readvariableop_resource:	D
(conv2d_65_conv2d_readvariableop_resource:8
)conv2d_65_biasadd_readvariableop_resource:	D
(conv2d_66_conv2d_readvariableop_resource:8
)conv2d_66_biasadd_readvariableop_resource:	D
(conv2d_67_conv2d_readvariableop_resource:8
)conv2d_67_biasadd_readvariableop_resource:	D
(conv2d_68_conv2d_readvariableop_resource:8
)conv2d_68_biasadd_readvariableop_resource:	D
(conv2d_69_conv2d_readvariableop_resource:8
)conv2d_69_biasadd_readvariableop_resource:	D
(conv2d_70_conv2d_readvariableop_resource:8
)conv2d_70_biasadd_readvariableop_resource:	C
(conv2d_71_conv2d_readvariableop_resource:À@7
)conv2d_71_biasadd_readvariableop_resource:@B
(conv2d_72_conv2d_readvariableop_resource:@@7
)conv2d_72_biasadd_readvariableop_resource:@B
(conv2d_73_conv2d_readvariableop_resource:` 7
)conv2d_73_biasadd_readvariableop_resource: B
(conv2d_74_conv2d_readvariableop_resource:  7
)conv2d_74_biasadd_readvariableop_resource: B
(conv2d_75_conv2d_readvariableop_resource: 7
)conv2d_75_biasadd_readvariableop_resource:
identity¢ conv2d_57/BiasAdd/ReadVariableOp¢conv2d_57/Conv2D/ReadVariableOp¢ conv2d_58/BiasAdd/ReadVariableOp¢conv2d_58/Conv2D/ReadVariableOp¢ conv2d_59/BiasAdd/ReadVariableOp¢conv2d_59/Conv2D/ReadVariableOp¢ conv2d_60/BiasAdd/ReadVariableOp¢conv2d_60/Conv2D/ReadVariableOp¢ conv2d_61/BiasAdd/ReadVariableOp¢conv2d_61/Conv2D/ReadVariableOp¢ conv2d_62/BiasAdd/ReadVariableOp¢conv2d_62/Conv2D/ReadVariableOp¢ conv2d_63/BiasAdd/ReadVariableOp¢conv2d_63/Conv2D/ReadVariableOp¢ conv2d_64/BiasAdd/ReadVariableOp¢conv2d_64/Conv2D/ReadVariableOp¢ conv2d_65/BiasAdd/ReadVariableOp¢conv2d_65/Conv2D/ReadVariableOp¢ conv2d_66/BiasAdd/ReadVariableOp¢conv2d_66/Conv2D/ReadVariableOp¢ conv2d_67/BiasAdd/ReadVariableOp¢conv2d_67/Conv2D/ReadVariableOp¢ conv2d_68/BiasAdd/ReadVariableOp¢conv2d_68/Conv2D/ReadVariableOp¢ conv2d_69/BiasAdd/ReadVariableOp¢conv2d_69/Conv2D/ReadVariableOp¢ conv2d_70/BiasAdd/ReadVariableOp¢conv2d_70/Conv2D/ReadVariableOp¢ conv2d_71/BiasAdd/ReadVariableOp¢conv2d_71/Conv2D/ReadVariableOp¢ conv2d_72/BiasAdd/ReadVariableOp¢conv2d_72/Conv2D/ReadVariableOp¢ conv2d_73/BiasAdd/ReadVariableOp¢conv2d_73/Conv2D/ReadVariableOp¢ conv2d_74/BiasAdd/ReadVariableOp¢conv2d_74/Conv2D/ReadVariableOp¢ conv2d_75/BiasAdd/ReadVariableOp¢conv2d_75/Conv2D/ReadVariableOp³
conv2d_57/Conv2D/ReadVariableOpReadVariableOp(conv2d_57_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_57/Conv2D/ReadVariableOpÃ
conv2d_57/Conv2DConv2Dinputs'conv2d_57/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d_57/Conv2Dª
 conv2d_57/BiasAdd/ReadVariableOpReadVariableOp)conv2d_57_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_57/BiasAdd/ReadVariableOp²
conv2d_57/BiasAddBiasAddconv2d_57/Conv2D:output:0(conv2d_57/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_57/BiasAdd
conv2d_57/ReluReluconv2d_57/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_57/Relu³
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_58/Conv2D/ReadVariableOpÙ
conv2d_58/Conv2DConv2Dconv2d_57/Relu:activations:0'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d_58/Conv2Dª
 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_58/BiasAdd/ReadVariableOp²
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_58/BiasAdd
conv2d_58/ReluReluconv2d_58/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_58/ReluÊ
max_pooling2d_12/MaxPoolMaxPoolconv2d_58/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPool³
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_59/Conv2D/ReadVariableOpÜ
conv2d_59/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
conv2d_59/Conv2Dª
 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_59/BiasAdd/ReadVariableOp°
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_59/BiasAdd~
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_59/Relu³
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_60/Conv2D/ReadVariableOp×
conv2d_60/Conv2DConv2Dconv2d_59/Relu:activations:0'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
conv2d_60/Conv2Dª
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp°
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_60/BiasAdd~
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_60/ReluÊ
max_pooling2d_13/MaxPoolMaxPoolconv2d_60/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPool´
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_61/Conv2D/ReadVariableOpÝ
conv2d_61/Conv2DConv2D!max_pooling2d_13/MaxPool:output:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv2d_61/Conv2D«
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_61/BiasAdd/ReadVariableOp±
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_61/BiasAdd
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_61/Reluµ
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_62/Conv2D/ReadVariableOpØ
conv2d_62/Conv2DConv2Dconv2d_61/Relu:activations:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv2d_62/Conv2D«
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_62/BiasAdd/ReadVariableOp±
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_62/BiasAdd
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_62/ReluË
max_pooling2d_14/MaxPoolMaxPoolconv2d_62/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_14/MaxPoolµ
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_63/Conv2D/ReadVariableOpÝ
conv2d_63/Conv2DConv2D!max_pooling2d_14/MaxPool:output:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_63/Conv2D«
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_63/BiasAdd/ReadVariableOp±
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_63/BiasAdd
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_63/Reluµ
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_64/Conv2D/ReadVariableOpØ
conv2d_64/Conv2DConv2Dconv2d_63/Relu:activations:0'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_64/Conv2D«
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_64/BiasAdd/ReadVariableOp±
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_64/BiasAdd
conv2d_64/ReluReluconv2d_64/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_64/ReluË
max_pooling2d_15/MaxPoolMaxPoolconv2d_64/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPoolµ
conv2d_65/Conv2D/ReadVariableOpReadVariableOp(conv2d_65_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_65/Conv2D/ReadVariableOpÝ
conv2d_65/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_65/Conv2D«
 conv2d_65/BiasAdd/ReadVariableOpReadVariableOp)conv2d_65_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_65/BiasAdd/ReadVariableOp±
conv2d_65/BiasAddBiasAddconv2d_65/Conv2D:output:0(conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_65/BiasAdd
conv2d_65/ReluReluconv2d_65/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_65/Reluµ
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_66/Conv2D/ReadVariableOpØ
conv2d_66/Conv2DConv2Dconv2d_65/Relu:activations:0'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_66/Conv2D«
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_66/BiasAdd/ReadVariableOp±
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_66/BiasAdd
conv2d_66/ReluReluconv2d_66/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_66/Relu
up_sampling2d_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_12/Const
up_sampling2d_12/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_12/Const_1
up_sampling2d_12/mulMulup_sampling2d_12/Const:output:0!up_sampling2d_12/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_12/mul
-up_sampling2d_12/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_66/Relu:activations:0up_sampling2d_12/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2/
-up_sampling2d_12/resize/ResizeNearestNeighborz
concatenate_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_12/concat/axis
concatenate_12/concatConcatV2>up_sampling2d_12/resize/ResizeNearestNeighbor:resized_images:0conv2d_64/Relu:activations:0#concatenate_12/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatenate_12/concatµ
conv2d_67/Conv2D/ReadVariableOpReadVariableOp(conv2d_67_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_67/Conv2D/ReadVariableOpÚ
conv2d_67/Conv2DConv2Dconcatenate_12/concat:output:0'conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_67/Conv2D«
 conv2d_67/BiasAdd/ReadVariableOpReadVariableOp)conv2d_67_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_67/BiasAdd/ReadVariableOp±
conv2d_67/BiasAddBiasAddconv2d_67/Conv2D:output:0(conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_67/BiasAdd
conv2d_67/ReluReluconv2d_67/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_67/Reluµ
conv2d_68/Conv2D/ReadVariableOpReadVariableOp(conv2d_68_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_68/Conv2D/ReadVariableOpØ
conv2d_68/Conv2DConv2Dconv2d_67/Relu:activations:0'conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_68/Conv2D«
 conv2d_68/BiasAdd/ReadVariableOpReadVariableOp)conv2d_68_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_68/BiasAdd/ReadVariableOp±
conv2d_68/BiasAddBiasAddconv2d_68/Conv2D:output:0(conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_68/BiasAdd
conv2d_68/ReluReluconv2d_68/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_68/Relu
up_sampling2d_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_13/Const
up_sampling2d_13/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_13/Const_1
up_sampling2d_13/mulMulup_sampling2d_13/Const:output:0!up_sampling2d_13/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_13/mul
-up_sampling2d_13/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_68/Relu:activations:0up_sampling2d_13/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(2/
-up_sampling2d_13/resize/ResizeNearestNeighborz
concatenate_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_13/concat/axis
concatenate_13/concatConcatV2>up_sampling2d_13/resize/ResizeNearestNeighbor:resized_images:0conv2d_62/Relu:activations:0#concatenate_13/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
concatenate_13/concatµ
conv2d_69/Conv2D/ReadVariableOpReadVariableOp(conv2d_69_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_69/Conv2D/ReadVariableOpÚ
conv2d_69/Conv2DConv2Dconcatenate_13/concat:output:0'conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv2d_69/Conv2D«
 conv2d_69/BiasAdd/ReadVariableOpReadVariableOp)conv2d_69_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_69/BiasAdd/ReadVariableOp±
conv2d_69/BiasAddBiasAddconv2d_69/Conv2D:output:0(conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_69/BiasAdd
conv2d_69/ReluReluconv2d_69/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_69/Reluµ
conv2d_70/Conv2D/ReadVariableOpReadVariableOp(conv2d_70_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_70/Conv2D/ReadVariableOpØ
conv2d_70/Conv2DConv2Dconv2d_69/Relu:activations:0'conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
2
conv2d_70/Conv2D«
 conv2d_70/BiasAdd/ReadVariableOpReadVariableOp)conv2d_70_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_70/BiasAdd/ReadVariableOp±
conv2d_70/BiasAddBiasAddconv2d_70/Conv2D:output:0(conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_70/BiasAdd
conv2d_70/ReluReluconv2d_70/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
conv2d_70/Relu
up_sampling2d_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
up_sampling2d_14/Const
up_sampling2d_14/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_14/Const_1
up_sampling2d_14/mulMulup_sampling2d_14/Const:output:0!up_sampling2d_14/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_14/mul
-up_sampling2d_14/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_70/Relu:activations:0up_sampling2d_14/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
half_pixel_centers(2/
-up_sampling2d_14/resize/ResizeNearestNeighborz
concatenate_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_14/concat/axis
concatenate_14/concatConcatV2>up_sampling2d_14/resize/ResizeNearestNeighbor:resized_images:0conv2d_60/Relu:activations:0#concatenate_14/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@À2
concatenate_14/concat´
conv2d_71/Conv2D/ReadVariableOpReadVariableOp(conv2d_71_conv2d_readvariableop_resource*'
_output_shapes
:À@*
dtype02!
conv2d_71/Conv2D/ReadVariableOpÙ
conv2d_71/Conv2DConv2Dconcatenate_14/concat:output:0'conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
conv2d_71/Conv2Dª
 conv2d_71/BiasAdd/ReadVariableOpReadVariableOp)conv2d_71_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_71/BiasAdd/ReadVariableOp°
conv2d_71/BiasAddBiasAddconv2d_71/Conv2D:output:0(conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_71/BiasAdd~
conv2d_71/ReluReluconv2d_71/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_71/Relu³
conv2d_72/Conv2D/ReadVariableOpReadVariableOp(conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_72/Conv2D/ReadVariableOp×
conv2d_72/Conv2DConv2Dconv2d_71/Relu:activations:0'conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
paddingSAME*
strides
2
conv2d_72/Conv2Dª
 conv2d_72/BiasAdd/ReadVariableOpReadVariableOp)conv2d_72_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_72/BiasAdd/ReadVariableOp°
conv2d_72/BiasAddBiasAddconv2d_72/Conv2D:output:0(conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_72/BiasAdd~
conv2d_72/ReluReluconv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2
conv2d_72/Relu
up_sampling2d_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   2
up_sampling2d_15/Const
up_sampling2d_15/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_15/Const_1
up_sampling2d_15/mulMulup_sampling2d_15/Const:output:0!up_sampling2d_15/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_15/mul
-up_sampling2d_15/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_72/Relu:activations:0up_sampling2d_15/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2/
-up_sampling2d_15/resize/ResizeNearestNeighborz
concatenate_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_15/concat/axis
concatenate_15/concatConcatV2>up_sampling2d_15/resize/ResizeNearestNeighbor:resized_images:0conv2d_58/Relu:activations:0#concatenate_15/concat/axis:output:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
concatenate_15/concat³
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02!
conv2d_73/Conv2D/ReadVariableOpÛ
conv2d_73/Conv2DConv2Dconcatenate_15/concat:output:0'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d_73/Conv2Dª
 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_73/BiasAdd/ReadVariableOp²
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_73/BiasAdd
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_73/Relu³
conv2d_74/Conv2D/ReadVariableOpReadVariableOp(conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_74/Conv2D/ReadVariableOpÙ
conv2d_74/Conv2DConv2Dconv2d_73/Relu:activations:0'conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d_74/Conv2Dª
 conv2d_74/BiasAdd/ReadVariableOpReadVariableOp)conv2d_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_74/BiasAdd/ReadVariableOp²
conv2d_74/BiasAddBiasAddconv2d_74/Conv2D:output:0(conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_74/BiasAdd
conv2d_74/ReluReluconv2d_74/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_74/Relu³
conv2d_75/Conv2D/ReadVariableOpReadVariableOp(conv2d_75_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_75/Conv2D/ReadVariableOpÙ
conv2d_75/Conv2DConv2Dconv2d_74/Relu:activations:0'conv2d_75/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_75/Conv2Dª
 conv2d_75/BiasAdd/ReadVariableOpReadVariableOp)conv2d_75_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_75/BiasAdd/ReadVariableOp²
conv2d_75/BiasAddBiasAddconv2d_75/Conv2D:output:0(conv2d_75/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_75/BiasAdd
conv2d_75/SigmoidSigmoidconv2d_75/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_75/Sigmoid
IdentityIdentityconv2d_75/Sigmoid:y:0!^conv2d_57/BiasAdd/ReadVariableOp ^conv2d_57/Conv2D/ReadVariableOp!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp!^conv2d_65/BiasAdd/ReadVariableOp ^conv2d_65/Conv2D/ReadVariableOp!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOp!^conv2d_67/BiasAdd/ReadVariableOp ^conv2d_67/Conv2D/ReadVariableOp!^conv2d_68/BiasAdd/ReadVariableOp ^conv2d_68/Conv2D/ReadVariableOp!^conv2d_69/BiasAdd/ReadVariableOp ^conv2d_69/Conv2D/ReadVariableOp!^conv2d_70/BiasAdd/ReadVariableOp ^conv2d_70/Conv2D/ReadVariableOp!^conv2d_71/BiasAdd/ReadVariableOp ^conv2d_71/Conv2D/ReadVariableOp!^conv2d_72/BiasAdd/ReadVariableOp ^conv2d_72/Conv2D/ReadVariableOp!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp!^conv2d_74/BiasAdd/ReadVariableOp ^conv2d_74/Conv2D/ReadVariableOp!^conv2d_75/BiasAdd/ReadVariableOp ^conv2d_75/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*|
_input_shapesk
i:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_57/BiasAdd/ReadVariableOp conv2d_57/BiasAdd/ReadVariableOp2B
conv2d_57/Conv2D/ReadVariableOpconv2d_57/Conv2D/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp2D
 conv2d_65/BiasAdd/ReadVariableOp conv2d_65/BiasAdd/ReadVariableOp2B
conv2d_65/Conv2D/ReadVariableOpconv2d_65/Conv2D/ReadVariableOp2D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2D
 conv2d_67/BiasAdd/ReadVariableOp conv2d_67/BiasAdd/ReadVariableOp2B
conv2d_67/Conv2D/ReadVariableOpconv2d_67/Conv2D/ReadVariableOp2D
 conv2d_68/BiasAdd/ReadVariableOp conv2d_68/BiasAdd/ReadVariableOp2B
conv2d_68/Conv2D/ReadVariableOpconv2d_68/Conv2D/ReadVariableOp2D
 conv2d_69/BiasAdd/ReadVariableOp conv2d_69/BiasAdd/ReadVariableOp2B
conv2d_69/Conv2D/ReadVariableOpconv2d_69/Conv2D/ReadVariableOp2D
 conv2d_70/BiasAdd/ReadVariableOp conv2d_70/BiasAdd/ReadVariableOp2B
conv2d_70/Conv2D/ReadVariableOpconv2d_70/Conv2D/ReadVariableOp2D
 conv2d_71/BiasAdd/ReadVariableOp conv2d_71/BiasAdd/ReadVariableOp2B
conv2d_71/Conv2D/ReadVariableOpconv2d_71/Conv2D/ReadVariableOp2D
 conv2d_72/BiasAdd/ReadVariableOp conv2d_72/BiasAdd/ReadVariableOp2B
conv2d_72/Conv2D/ReadVariableOpconv2d_72/Conv2D/ReadVariableOp2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp2D
 conv2d_74/BiasAdd/ReadVariableOp conv2d_74/BiasAdd/ReadVariableOp2B
conv2d_74/Conv2D/ReadVariableOpconv2d_74/Conv2D/ReadVariableOp2D
 conv2d_75/BiasAdd/ReadVariableOp conv2d_75/BiasAdd/ReadVariableOp2B
conv2d_75/Conv2D/ReadVariableOpconv2d_75/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
¡
)__inference_conv2d_64_layer_call_fn_71071

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_692332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
u
I__inference_concatenate_15_layer_call_and_return_conditional_losses_71294
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ :k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1
 
ý
D__inference_conv2d_73_layer_call_and_return_conditional_losses_69427

inputs8
conv2d_readvariableop_resource:` -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
 
ý
D__inference_conv2d_57_layer_call_and_return_conditional_losses_70942

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu¡
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*À
serving_default¬
E
input_4:
serving_default_input_4:0ÿÿÿÿÿÿÿÿÿG
	conv2d_75:
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:×î
ÝÝ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer-20
layer_with_weights-12
layer-21
layer_with_weights-13
layer-22
layer-23
layer-24
layer_with_weights-14
layer-25
layer_with_weights-15
layer-26
layer-27
layer-28
layer_with_weights-16
layer-29
layer_with_weights-17
layer-30
 layer_with_weights-18
 layer-31
!	optimizer
"regularization_losses
#	variables
$trainable_variables
%	keras_api
&
signatures
Þ_default_save_signature
ß__call__
+à&call_and_return_all_conditional_losses"áÔ
_tf_keras_networkÄÔ{"name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_57", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_57", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_58", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_58", "inbound_nodes": [[["conv2d_57", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_12", "inbound_nodes": [[["conv2d_58", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_59", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_59", "inbound_nodes": [[["max_pooling2d_12", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_60", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_60", "inbound_nodes": [[["conv2d_59", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_13", "inbound_nodes": [[["conv2d_60", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_61", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_61", "inbound_nodes": [[["max_pooling2d_13", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_62", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_62", "inbound_nodes": [[["conv2d_61", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_14", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_14", "inbound_nodes": [[["conv2d_62", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_63", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_63", "inbound_nodes": [[["max_pooling2d_14", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_64", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_64", "inbound_nodes": [[["conv2d_63", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_15", "inbound_nodes": [[["conv2d_64", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_65", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_65", "inbound_nodes": [[["max_pooling2d_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_66", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_66", "inbound_nodes": [[["conv2d_65", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_12", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_12", "inbound_nodes": [[["conv2d_66", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_12", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_12", "inbound_nodes": [[["up_sampling2d_12", 0, 0, {}], ["conv2d_64", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_67", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_67", "inbound_nodes": [[["concatenate_12", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_68", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_68", "inbound_nodes": [[["conv2d_67", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_13", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_13", "inbound_nodes": [[["conv2d_68", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_13", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_13", "inbound_nodes": [[["up_sampling2d_13", 0, 0, {}], ["conv2d_62", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_69", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_69", "inbound_nodes": [[["concatenate_13", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_70", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_70", "inbound_nodes": [[["conv2d_69", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_14", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_14", "inbound_nodes": [[["conv2d_70", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_14", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_14", "inbound_nodes": [[["up_sampling2d_14", 0, 0, {}], ["conv2d_60", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_71", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_71", "inbound_nodes": [[["concatenate_14", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_72", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_72", "inbound_nodes": [[["conv2d_71", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_15", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_15", "inbound_nodes": [[["conv2d_72", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_15", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_15", "inbound_nodes": [[["up_sampling2d_15", 0, 0, {}], ["conv2d_58", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_73", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_73", "inbound_nodes": [[["concatenate_15", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_74", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_74", "inbound_nodes": [[["conv2d_73", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_75", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_75", "inbound_nodes": [[["conv2d_74", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["conv2d_75", 0, 0]]}, "shared_object_id": 70, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128, 128, 3]}, "float32", "input_4"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_57", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_57", "inbound_nodes": [[["input_4", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv2d_58", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_58", "inbound_nodes": [[["conv2d_57", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_12", "inbound_nodes": [[["conv2d_58", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Conv2D", "config": {"name": "conv2d_59", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_59", "inbound_nodes": [[["max_pooling2d_12", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "conv2d_60", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_60", "inbound_nodes": [[["conv2d_59", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_13", "inbound_nodes": [[["conv2d_60", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Conv2D", "config": {"name": "conv2d_61", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_61", "inbound_nodes": [[["max_pooling2d_13", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_62", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_62", "inbound_nodes": [[["conv2d_61", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_14", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_14", "inbound_nodes": [[["conv2d_62", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "conv2d_63", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_63", "inbound_nodes": [[["max_pooling2d_14", 0, 0, {}]]], "shared_object_id": 24}, {"class_name": "Conv2D", "config": {"name": "conv2d_64", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_64", "inbound_nodes": [[["conv2d_63", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_15", "inbound_nodes": [[["conv2d_64", 0, 0, {}]]], "shared_object_id": 28}, {"class_name": "Conv2D", "config": {"name": "conv2d_65", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_65", "inbound_nodes": [[["max_pooling2d_15", 0, 0, {}]]], "shared_object_id": 31}, {"class_name": "Conv2D", "config": {"name": "conv2d_66", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_66", "inbound_nodes": [[["conv2d_65", 0, 0, {}]]], "shared_object_id": 34}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_12", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_12", "inbound_nodes": [[["conv2d_66", 0, 0, {}]]], "shared_object_id": 35}, {"class_name": "Concatenate", "config": {"name": "concatenate_12", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_12", "inbound_nodes": [[["up_sampling2d_12", 0, 0, {}], ["conv2d_64", 0, 0, {}]]], "shared_object_id": 36}, {"class_name": "Conv2D", "config": {"name": "conv2d_67", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_67", "inbound_nodes": [[["concatenate_12", 0, 0, {}]]], "shared_object_id": 39}, {"class_name": "Conv2D", "config": {"name": "conv2d_68", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_68", "inbound_nodes": [[["conv2d_67", 0, 0, {}]]], "shared_object_id": 42}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_13", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_13", "inbound_nodes": [[["conv2d_68", 0, 0, {}]]], "shared_object_id": 43}, {"class_name": "Concatenate", "config": {"name": "concatenate_13", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_13", "inbound_nodes": [[["up_sampling2d_13", 0, 0, {}], ["conv2d_62", 0, 0, {}]]], "shared_object_id": 44}, {"class_name": "Conv2D", "config": {"name": "conv2d_69", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_69", "inbound_nodes": [[["concatenate_13", 0, 0, {}]]], "shared_object_id": 47}, {"class_name": "Conv2D", "config": {"name": "conv2d_70", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 48}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 49}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_70", "inbound_nodes": [[["conv2d_69", 0, 0, {}]]], "shared_object_id": 50}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_14", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_14", "inbound_nodes": [[["conv2d_70", 0, 0, {}]]], "shared_object_id": 51}, {"class_name": "Concatenate", "config": {"name": "concatenate_14", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_14", "inbound_nodes": [[["up_sampling2d_14", 0, 0, {}], ["conv2d_60", 0, 0, {}]]], "shared_object_id": 52}, {"class_name": "Conv2D", "config": {"name": "conv2d_71", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_71", "inbound_nodes": [[["concatenate_14", 0, 0, {}]]], "shared_object_id": 55}, {"class_name": "Conv2D", "config": {"name": "conv2d_72", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 56}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_72", "inbound_nodes": [[["conv2d_71", 0, 0, {}]]], "shared_object_id": 58}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_15", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_15", "inbound_nodes": [[["conv2d_72", 0, 0, {}]]], "shared_object_id": 59}, {"class_name": "Concatenate", "config": {"name": "concatenate_15", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_15", "inbound_nodes": [[["up_sampling2d_15", 0, 0, {}], ["conv2d_58", 0, 0, {}]]], "shared_object_id": 60}, {"class_name": "Conv2D", "config": {"name": "conv2d_73", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_73", "inbound_nodes": [[["concatenate_15", 0, 0, {}]]], "shared_object_id": 63}, {"class_name": "Conv2D", "config": {"name": "conv2d_74", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 64}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 65}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_74", "inbound_nodes": [[["conv2d_73", 0, 0, {}]]], "shared_object_id": 66}, {"class_name": "Conv2D", "config": {"name": "conv2d_75", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 67}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_75", "inbound_nodes": [[["conv2d_74", 0, 0, {}]]], "shared_object_id": 69}], "input_layers": [["input_4", 0, 0]], "output_layers": [["conv2d_75", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "IoU", "dtype": "float32", "fn": "IoU"}, "shared_object_id": 72}, {"class_name": "BinaryAccuracy", "config": {"name": "accuracy", "dtype": "float32", "threshold": 0.5}, "shared_object_id": 73}, {"class_name": "AUC", "config": {"name": "ROC_AUC", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}, "shared_object_id": 74}, {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 75}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 76}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ý"ú
_tf_keras_input_layerÚ{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
ÿ


'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
á__call__
+â&call_and_return_all_conditional_losses"Ø	
_tf_keras_layer¾	{"name": "conv2d_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_57", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_4", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}}


-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"Ü	
_tf_keras_layerÂ	{"name": "conv2d_58", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_58", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_57", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 78}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 32]}}
à
3regularization_losses
4	variables
5trainable_variables
6	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"name": "max_pooling2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_58", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 79}}


7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
ç__call__
+è&call_and_return_all_conditional_losses"â	
_tf_keras_layerÈ	{"name": "conv2d_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_59", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["max_pooling2d_12", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 80}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}


=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
é__call__
+ê&call_and_return_all_conditional_losses"Ý	
_tf_keras_layerÃ	{"name": "conv2d_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_60", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_59", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 81}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
á
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"name": "max_pooling2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_60", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 82}}


Gkernel
Hbias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
í__call__
+î&call_and_return_all_conditional_losses"å	
_tf_keras_layerË	{"name": "conv2d_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_61", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["max_pooling2d_13", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 83}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 64]}}


Mkernel
Nbias
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
ï__call__
+ð&call_and_return_all_conditional_losses"à	
_tf_keras_layerÆ	{"name": "conv2d_62", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_62", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_61", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 84}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
á
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"name": "max_pooling2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_14", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_62", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 85}}


Wkernel
Xbias
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"ç	
_tf_keras_layerÍ	{"name": "conv2d_63", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_63", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["max_pooling2d_14", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 86}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}


]kernel
^bias
_regularization_losses
`	variables
atrainable_variables
b	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"à	
_tf_keras_layerÆ	{"name": "conv2d_64", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_64", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_63", 0, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 87}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 256]}}
á
cregularization_losses
d	variables
etrainable_variables
f	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"name": "max_pooling2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_64", 0, 0, {}]]], "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 88}}


gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"å	
_tf_keras_layerË	{"name": "conv2d_65", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_65", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["max_pooling2d_15", 0, 0, {}]]], "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 89}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 256]}}


mkernel
nbias
oregularization_losses
p	variables
qtrainable_variables
r	keras_api
û__call__
+ü&call_and_return_all_conditional_losses"Þ	
_tf_keras_layerÄ	{"name": "conv2d_66", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_66", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_65", 0, 0, {}]]], "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}, "shared_object_id": 90}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 512]}}
«
sregularization_losses
t	variables
utrainable_variables
v	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses"
_tf_keras_layer{"name": "up_sampling2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_12", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "inbound_nodes": [[["conv2d_66", 0, 0, {}]]], "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 91}}
É
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"¸
_tf_keras_layer{"name": "concatenate_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_12", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["up_sampling2d_12", 0, 0, {}], ["conv2d_64", 0, 0, {}]]], "shared_object_id": 36, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16, 16, 512]}, {"class_name": "TensorShape", "items": [null, 16, 16, 256]}]}


{kernel
|bias
}regularization_losses
~	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"å	
_tf_keras_layerË	{"name": "conv2d_67", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_67", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_12", 0, 0, {}]]], "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 768}}, "shared_object_id": 92}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 768]}}

kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"à	
_tf_keras_layerÆ	{"name": "conv2d_68", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_68", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_67", 0, 0, {}]]], "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 93}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 256]}}
¯
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"name": "up_sampling2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_13", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "inbound_nodes": [[["conv2d_68", 0, 0, {}]]], "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 94}}
Í
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"¸
_tf_keras_layer{"name": "concatenate_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_13", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["up_sampling2d_13", 0, 0, {}], ["conv2d_62", 0, 0, {}]]], "shared_object_id": 44, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 256]}, {"class_name": "TensorShape", "items": [null, 32, 32, 128]}]}

kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"å	
_tf_keras_layerË	{"name": "conv2d_69", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_69", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_13", 0, 0, {}]]], "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 384}}, "shared_object_id": 95}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 384]}}

kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"à	
_tf_keras_layerÆ	{"name": "conv2d_70", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_70", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 48}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 49}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_69", 0, 0, {}]]], "shared_object_id": 50, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 96}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 128]}}
¯
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"name": "up_sampling2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_14", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "inbound_nodes": [[["conv2d_70", 0, 0, {}]]], "shared_object_id": 51, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 97}}
Ì
regularization_losses
 	variables
¡trainable_variables
¢	keras_api
__call__
+&call_and_return_all_conditional_losses"·
_tf_keras_layer{"name": "concatenate_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_14", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["up_sampling2d_14", 0, 0, {}], ["conv2d_60", 0, 0, {}]]], "shared_object_id": 52, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64, 64, 128]}, {"class_name": "TensorShape", "items": [null, 64, 64, 64]}]}

£kernel
	¤bias
¥regularization_losses
¦	variables
§trainable_variables
¨	keras_api
__call__
+&call_and_return_all_conditional_losses"ä	
_tf_keras_layerÊ	{"name": "conv2d_71", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_71", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_14", 0, 0, {}]]], "shared_object_id": 55, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 192}}, "shared_object_id": 98}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 192]}}

©kernel
	ªbias
«regularization_losses
¬	variables
­trainable_variables
®	keras_api
__call__
+&call_and_return_all_conditional_losses"Ý	
_tf_keras_layerÃ	{"name": "conv2d_72", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_72", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 56}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_71", 0, 0, {}]]], "shared_object_id": 58, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 99}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 64]}}
°
¯regularization_losses
°	variables
±trainable_variables
²	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"name": "up_sampling2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_15", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "inbound_nodes": [[["conv2d_72", 0, 0, {}]]], "shared_object_id": 59, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 100}}
Ï
³regularization_losses
´	variables
µtrainable_variables
¶	keras_api
__call__
+&call_and_return_all_conditional_losses"º
_tf_keras_layer {"name": "concatenate_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_15", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["up_sampling2d_15", 0, 0, {}], ["conv2d_58", 0, 0, {}]]], "shared_object_id": 60, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 128, 64]}, {"class_name": "TensorShape", "items": [null, 128, 128, 32]}]}

·kernel
	¸bias
¹regularization_losses
º	variables
»trainable_variables
¼	keras_api
__call__
+&call_and_return_all_conditional_losses"å	
_tf_keras_layerË	{"name": "conv2d_73", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_73", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_15", 0, 0, {}]]], "shared_object_id": 63, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 96}}, "shared_object_id": 101}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 96]}}

½kernel
	¾bias
¿regularization_losses
À	variables
Átrainable_variables
Â	keras_api
__call__
+&call_and_return_all_conditional_losses"à	
_tf_keras_layerÆ	{"name": "conv2d_74", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_74", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 64}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 65}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_73", 0, 0, {}]]], "shared_object_id": 66, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 102}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 32]}}

Ãkernel
	Äbias
Åregularization_losses
Æ	variables
Çtrainable_variables
È	keras_api
__call__
+&call_and_return_all_conditional_losses"â	
_tf_keras_layerÈ	{"name": "conv2d_75", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_75", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 67}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_74", 0, 0, {}]]], "shared_object_id": 69, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 103}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 32]}}
ð
	Éiter
Êbeta_1
Ëbeta_2

Ìdecay
Ílearning_rate'm(m-m.m7m8m=m>mGmHmMmNmWmXm]m ^m¡gm¢hm£mm¤nm¥{m¦|m§	m¨	m©	mª	m«	m¬	m­	£m®	¤m¯	©m°	ªm±	·m²	¸m³	½m´	¾mµ	Ãm¶	Äm·'v¸(v¹-vº.v»7v¼8v½=v¾>v¿GvÀHvÁMvÂNvÃWvÄXvÅ]vÆ^vÇgvÈhvÉmvÊnvË{vÌ|vÍ	vÎ	vÏ	vÐ	vÑ	vÒ	vÓ	£vÔ	¤vÕ	©vÖ	ªv×	·vØ	¸vÙ	½vÚ	¾vÛ	ÃvÜ	ÄvÝ"
	optimizer
 "
trackable_list_wrapper
Ö
'0
(1
-2
.3
74
85
=6
>7
G8
H9
M10
N11
W12
X13
]14
^15
g16
h17
m18
n19
{20
|21
22
23
24
25
26
27
£28
¤29
©30
ª31
·32
¸33
½34
¾35
Ã36
Ä37"
trackable_list_wrapper
Ö
'0
(1
-2
.3
74
85
=6
>7
G8
H9
M10
N11
W12
X13
]14
^15
g16
h17
m18
n19
{20
|21
22
23
24
25
26
27
£28
¤29
©30
ª31
·32
¸33
½34
¾35
Ã36
Ä37"
trackable_list_wrapper
Ó
Înon_trainable_variables
Ïlayer_metrics
"regularization_losses
#	variables
Ðlayers
$trainable_variables
 Ñlayer_regularization_losses
Òmetrics
ß__call__
Þ_default_save_signature
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
*:( 2conv2d_57/kernel
: 2conv2d_57/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
µ
Ónon_trainable_variables
Ôlayer_metrics
)regularization_losses
*	variables
Õlayers
+trainable_variables
 Ölayer_regularization_losses
×metrics
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_58/kernel
: 2conv2d_58/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
µ
Ønon_trainable_variables
Ùlayer_metrics
/regularization_losses
0	variables
Úlayers
1trainable_variables
 Ûlayer_regularization_losses
Ümetrics
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ýnon_trainable_variables
Þlayer_metrics
3regularization_losses
4	variables
ßlayers
5trainable_variables
 àlayer_regularization_losses
ámetrics
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_59/kernel
:@2conv2d_59/bias
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
µ
ânon_trainable_variables
ãlayer_metrics
9regularization_losses
:	variables
älayers
;trainable_variables
 ålayer_regularization_losses
æmetrics
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_60/kernel
:@2conv2d_60/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
µ
çnon_trainable_variables
èlayer_metrics
?regularization_losses
@	variables
élayers
Atrainable_variables
 êlayer_regularization_losses
ëmetrics
é__call__
+ê&call_and_return_all_conditional_losses
'ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ìnon_trainable_variables
ílayer_metrics
Cregularization_losses
D	variables
îlayers
Etrainable_variables
 ïlayer_regularization_losses
ðmetrics
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
+:)@2conv2d_61/kernel
:2conv2d_61/bias
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
µ
ñnon_trainable_variables
òlayer_metrics
Iregularization_losses
J	variables
ólayers
Ktrainable_variables
 ôlayer_regularization_losses
õmetrics
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_62/kernel
:2conv2d_62/bias
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
µ
önon_trainable_variables
÷layer_metrics
Oregularization_losses
P	variables
ølayers
Qtrainable_variables
 ùlayer_regularization_losses
úmetrics
ï__call__
+ð&call_and_return_all_conditional_losses
'ð"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ûnon_trainable_variables
ülayer_metrics
Sregularization_losses
T	variables
ýlayers
Utrainable_variables
 þlayer_regularization_losses
ÿmetrics
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_63/kernel
:2conv2d_63/bias
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
Yregularization_losses
Z	variables
layers
[trainable_variables
 layer_regularization_losses
metrics
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_64/kernel
:2conv2d_64/bias
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
_regularization_losses
`	variables
layers
atrainable_variables
 layer_regularization_losses
metrics
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
cregularization_losses
d	variables
layers
etrainable_variables
 layer_regularization_losses
metrics
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_65/kernel
:2conv2d_65/bias
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
iregularization_losses
j	variables
layers
ktrainable_variables
 layer_regularization_losses
metrics
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_66/kernel
:2conv2d_66/bias
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
oregularization_losses
p	variables
layers
qtrainable_variables
 layer_regularization_losses
metrics
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
sregularization_losses
t	variables
layers
utrainable_variables
 layer_regularization_losses
metrics
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layer_metrics
wregularization_losses
x	variables
 layers
ytrainable_variables
 ¡layer_regularization_losses
¢metrics
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_67/kernel
:2conv2d_67/bias
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
µ
£non_trainable_variables
¤layer_metrics
}regularization_losses
~	variables
¥layers
trainable_variables
 ¦layer_regularization_losses
§metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_68/kernel
:2conv2d_68/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
¨non_trainable_variables
©layer_metrics
regularization_losses
	variables
ªlayers
trainable_variables
 «layer_regularization_losses
¬metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
­non_trainable_variables
®layer_metrics
regularization_losses
	variables
¯layers
trainable_variables
 °layer_regularization_losses
±metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
²non_trainable_variables
³layer_metrics
regularization_losses
	variables
´layers
trainable_variables
 µlayer_regularization_losses
¶metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_69/kernel
:2conv2d_69/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
·non_trainable_variables
¸layer_metrics
regularization_losses
	variables
¹layers
trainable_variables
 ºlayer_regularization_losses
»metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_70/kernel
:2conv2d_70/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
¼non_trainable_variables
½layer_metrics
regularization_losses
	variables
¾layers
trainable_variables
 ¿layer_regularization_losses
Àmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ánon_trainable_variables
Âlayer_metrics
regularization_losses
	variables
Ãlayers
trainable_variables
 Älayer_regularization_losses
Åmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ænon_trainable_variables
Çlayer_metrics
regularization_losses
 	variables
Èlayers
¡trainable_variables
 Élayer_regularization_losses
Êmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)À@2conv2d_71/kernel
:@2conv2d_71/bias
 "
trackable_list_wrapper
0
£0
¤1"
trackable_list_wrapper
0
£0
¤1"
trackable_list_wrapper
¸
Ënon_trainable_variables
Ìlayer_metrics
¥regularization_losses
¦	variables
Ílayers
§trainable_variables
 Îlayer_regularization_losses
Ïmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_72/kernel
:@2conv2d_72/bias
 "
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
0
©0
ª1"
trackable_list_wrapper
¸
Ðnon_trainable_variables
Ñlayer_metrics
«regularization_losses
¬	variables
Òlayers
­trainable_variables
 Ólayer_regularization_losses
Ômetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Õnon_trainable_variables
Ölayer_metrics
¯regularization_losses
°	variables
×layers
±trainable_variables
 Ølayer_regularization_losses
Ùmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Únon_trainable_variables
Ûlayer_metrics
³regularization_losses
´	variables
Ülayers
µtrainable_variables
 Ýlayer_regularization_losses
Þmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:(` 2conv2d_73/kernel
: 2conv2d_73/bias
 "
trackable_list_wrapper
0
·0
¸1"
trackable_list_wrapper
0
·0
¸1"
trackable_list_wrapper
¸
ßnon_trainable_variables
àlayer_metrics
¹regularization_losses
º	variables
álayers
»trainable_variables
 âlayer_regularization_losses
ãmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_74/kernel
: 2conv2d_74/bias
 "
trackable_list_wrapper
0
½0
¾1"
trackable_list_wrapper
0
½0
¾1"
trackable_list_wrapper
¸
änon_trainable_variables
ålayer_metrics
¿regularization_losses
À	variables
ælayers
Átrainable_variables
 çlayer_regularization_losses
èmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_75/kernel
:2conv2d_75/bias
 "
trackable_list_wrapper
0
Ã0
Ä1"
trackable_list_wrapper
0
Ã0
Ä1"
trackable_list_wrapper
¸
énon_trainable_variables
êlayer_metrics
Åregularization_losses
Æ	variables
ëlayers
Çtrainable_variables
 ìlayer_regularization_losses
ímetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31"
trackable_list_wrapper
 "
trackable_list_wrapper
P
î0
ï1
ð2
ñ3
ò4
ó5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ù

ôtotal

õcount
ö	variables
÷	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 104}


øtotal

ùcount
ú
_fn_kwargs
û	variables
ü	keras_api"µ
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "IoU", "dtype": "float32", "config": {"name": "IoU", "dtype": "float32", "fn": "IoU"}, "shared_object_id": 72}


ýtotal

þcount
ÿ
_fn_kwargs
	variables
	keras_api"Á
_tf_keras_metric¦{"class_name": "BinaryAccuracy", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "threshold": 0.5}, "shared_object_id": 73}
Õ"
true_positives
true_negatives
false_positives
false_negatives
	variables
	keras_api"Ü!
_tf_keras_metricÁ!{"class_name": "AUC", "name": "ROC_AUC", "dtype": "float32", "config": {"name": "ROC_AUC", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}, "shared_object_id": 74}
À

thresholds
true_positives
false_positives
	variables
	keras_api"á
_tf_keras_metricÆ{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 75}
·

thresholds
true_positives
false_negatives
	variables
	keras_api"Ø
_tf_keras_metric½{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "shared_object_id": 76}
:  (2total
:  (2count
0
ô0
õ1"
trackable_list_wrapper
.
ö	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ø0
ù1"
trackable_list_wrapper
.
û	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ý0
þ1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:È (2true_positives
:È (2true_negatives
 :È (2false_positives
 :È (2false_negatives
@
0
1
2
3"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
/:- 2Adam/conv2d_57/kernel/m
!: 2Adam/conv2d_57/bias/m
/:-  2Adam/conv2d_58/kernel/m
!: 2Adam/conv2d_58/bias/m
/:- @2Adam/conv2d_59/kernel/m
!:@2Adam/conv2d_59/bias/m
/:-@@2Adam/conv2d_60/kernel/m
!:@2Adam/conv2d_60/bias/m
0:.@2Adam/conv2d_61/kernel/m
": 2Adam/conv2d_61/bias/m
1:/2Adam/conv2d_62/kernel/m
": 2Adam/conv2d_62/bias/m
1:/2Adam/conv2d_63/kernel/m
": 2Adam/conv2d_63/bias/m
1:/2Adam/conv2d_64/kernel/m
": 2Adam/conv2d_64/bias/m
1:/2Adam/conv2d_65/kernel/m
": 2Adam/conv2d_65/bias/m
1:/2Adam/conv2d_66/kernel/m
": 2Adam/conv2d_66/bias/m
1:/2Adam/conv2d_67/kernel/m
": 2Adam/conv2d_67/bias/m
1:/2Adam/conv2d_68/kernel/m
": 2Adam/conv2d_68/bias/m
1:/2Adam/conv2d_69/kernel/m
": 2Adam/conv2d_69/bias/m
1:/2Adam/conv2d_70/kernel/m
": 2Adam/conv2d_70/bias/m
0:.À@2Adam/conv2d_71/kernel/m
!:@2Adam/conv2d_71/bias/m
/:-@@2Adam/conv2d_72/kernel/m
!:@2Adam/conv2d_72/bias/m
/:-` 2Adam/conv2d_73/kernel/m
!: 2Adam/conv2d_73/bias/m
/:-  2Adam/conv2d_74/kernel/m
!: 2Adam/conv2d_74/bias/m
/:- 2Adam/conv2d_75/kernel/m
!:2Adam/conv2d_75/bias/m
/:- 2Adam/conv2d_57/kernel/v
!: 2Adam/conv2d_57/bias/v
/:-  2Adam/conv2d_58/kernel/v
!: 2Adam/conv2d_58/bias/v
/:- @2Adam/conv2d_59/kernel/v
!:@2Adam/conv2d_59/bias/v
/:-@@2Adam/conv2d_60/kernel/v
!:@2Adam/conv2d_60/bias/v
0:.@2Adam/conv2d_61/kernel/v
": 2Adam/conv2d_61/bias/v
1:/2Adam/conv2d_62/kernel/v
": 2Adam/conv2d_62/bias/v
1:/2Adam/conv2d_63/kernel/v
": 2Adam/conv2d_63/bias/v
1:/2Adam/conv2d_64/kernel/v
": 2Adam/conv2d_64/bias/v
1:/2Adam/conv2d_65/kernel/v
": 2Adam/conv2d_65/bias/v
1:/2Adam/conv2d_66/kernel/v
": 2Adam/conv2d_66/bias/v
1:/2Adam/conv2d_67/kernel/v
": 2Adam/conv2d_67/bias/v
1:/2Adam/conv2d_68/kernel/v
": 2Adam/conv2d_68/bias/v
1:/2Adam/conv2d_69/kernel/v
": 2Adam/conv2d_69/bias/v
1:/2Adam/conv2d_70/kernel/v
": 2Adam/conv2d_70/bias/v
0:.À@2Adam/conv2d_71/kernel/v
!:@2Adam/conv2d_71/bias/v
/:-@@2Adam/conv2d_72/kernel/v
!:@2Adam/conv2d_72/bias/v
/:-` 2Adam/conv2d_73/kernel/v
!: 2Adam/conv2d_73/bias/v
/:-  2Adam/conv2d_74/kernel/v
!: 2Adam/conv2d_74/bias/v
/:- 2Adam/conv2d_75/kernel/v
!:2Adam/conv2d_75/bias/v
è2å
 __inference__wrapped_model_68969À
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *0¢-
+(
input_4ÿÿÿÿÿÿÿÿÿ
ê2ç
'__inference_model_3_layer_call_fn_69547
'__inference_model_3_layer_call_fn_70511
'__inference_model_3_layer_call_fn_70592
'__inference_model_3_layer_call_fn_70119À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
B__inference_model_3_layer_call_and_return_conditional_losses_70757
B__inference_model_3_layer_call_and_return_conditional_losses_70922
B__inference_model_3_layer_call_and_return_conditional_losses_70230
B__inference_model_3_layer_call_and_return_conditional_losses_70341À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_57_layer_call_fn_70931¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_57_layer_call_and_return_conditional_losses_70942¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_58_layer_call_fn_70951¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_58_layer_call_and_return_conditional_losses_70962¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_max_pooling2d_12_layer_call_fn_68981à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_68975à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ó2Ð
)__inference_conv2d_59_layer_call_fn_70971¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_59_layer_call_and_return_conditional_losses_70982¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_60_layer_call_fn_70991¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_60_layer_call_and_return_conditional_losses_71002¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_max_pooling2d_13_layer_call_fn_68993à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_68987à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ó2Ð
)__inference_conv2d_61_layer_call_fn_71011¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_61_layer_call_and_return_conditional_losses_71022¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_62_layer_call_fn_71031¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_62_layer_call_and_return_conditional_losses_71042¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_max_pooling2d_14_layer_call_fn_69005à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_68999à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ó2Ð
)__inference_conv2d_63_layer_call_fn_71051¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_63_layer_call_and_return_conditional_losses_71062¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_64_layer_call_fn_71071¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_64_layer_call_and_return_conditional_losses_71082¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_max_pooling2d_15_layer_call_fn_69017à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_69011à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ó2Ð
)__inference_conv2d_65_layer_call_fn_71091¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_65_layer_call_and_return_conditional_losses_71102¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_66_layer_call_fn_71111¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_66_layer_call_and_return_conditional_losses_71122¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_up_sampling2d_12_layer_call_fn_69036à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_69030à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ø2Õ
.__inference_concatenate_12_layer_call_fn_71128¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_concatenate_12_layer_call_and_return_conditional_losses_71135¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_67_layer_call_fn_71144¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_67_layer_call_and_return_conditional_losses_71155¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_68_layer_call_fn_71164¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_68_layer_call_and_return_conditional_losses_71175¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_up_sampling2d_13_layer_call_fn_69055à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_69049à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ø2Õ
.__inference_concatenate_13_layer_call_fn_71181¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_concatenate_13_layer_call_and_return_conditional_losses_71188¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_69_layer_call_fn_71197¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_69_layer_call_and_return_conditional_losses_71208¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_70_layer_call_fn_71217¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_70_layer_call_and_return_conditional_losses_71228¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_up_sampling2d_14_layer_call_fn_69074à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_69068à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ø2Õ
.__inference_concatenate_14_layer_call_fn_71234¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_concatenate_14_layer_call_and_return_conditional_losses_71241¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_71_layer_call_fn_71250¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_71_layer_call_and_return_conditional_losses_71261¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_72_layer_call_fn_71270¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_72_layer_call_and_return_conditional_losses_71281¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_up_sampling2d_15_layer_call_fn_69093à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
³2°
K__inference_up_sampling2d_15_layer_call_and_return_conditional_losses_69087à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ø2Õ
.__inference_concatenate_15_layer_call_fn_71287¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_concatenate_15_layer_call_and_return_conditional_losses_71294¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_73_layer_call_fn_71303¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_73_layer_call_and_return_conditional_losses_71314¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_74_layer_call_fn_71323¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_74_layer_call_and_return_conditional_losses_71334¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_75_layer_call_fn_71343¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_75_layer_call_and_return_conditional_losses_71354¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÊBÇ
#__inference_signature_wrapper_70430input_4"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 Ú
 __inference__wrapped_model_68969µ6'(-.78=>GHMNWX]^ghmn{|£¤©ª·¸½¾ÃÄ:¢7
0¢-
+(
input_4ÿÿÿÿÿÿÿÿÿ
ª "?ª<
:
	conv2d_75-*
	conv2d_75ÿÿÿÿÿÿÿÿÿþ
I__inference_concatenate_12_layer_call_and_return_conditional_losses_71135°~¢{
t¢q
ol
=:
inputs/0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+(
inputs/1ÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ö
.__inference_concatenate_12_layer_call_fn_71128£~¢{
t¢q
ol
=:
inputs/0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+(
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿþ
I__inference_concatenate_13_layer_call_and_return_conditional_losses_71188°~¢{
t¢q
ol
=:
inputs/0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+(
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 Ö
.__inference_concatenate_13_layer_call_fn_71181£~¢{
t¢q
ol
=:
inputs/0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+(
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª "!ÿÿÿÿÿÿÿÿÿ  ý
I__inference_concatenate_14_layer_call_and_return_conditional_losses_71241¯}¢z
s¢p
nk
=:
inputs/0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*'
inputs/1ÿÿÿÿÿÿÿÿÿ@@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@À
 Õ
.__inference_concatenate_14_layer_call_fn_71234¢}¢z
s¢p
nk
=:
inputs/0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*'
inputs/1ÿÿÿÿÿÿÿÿÿ@@@
ª "!ÿÿÿÿÿÿÿÿÿ@@Àÿ
I__inference_concatenate_15_layer_call_and_return_conditional_losses_71294±~¢{
t¢q
ol
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
,)
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ`
 ×
.__inference_concatenate_15_layer_call_fn_71287¤~¢{
t¢q
ol
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
,)
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ`¸
D__inference_conv2d_57_layer_call_and_return_conditional_losses_70942p'(9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_conv2d_57_layer_call_fn_70931c'(9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ ¸
D__inference_conv2d_58_layer_call_and_return_conditional_losses_70962p-.9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_conv2d_58_layer_call_fn_70951c-.9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ ´
D__inference_conv2d_59_layer_call_and_return_conditional_losses_70982l787¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
)__inference_conv2d_59_layer_call_fn_70971_787¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@ 
ª " ÿÿÿÿÿÿÿÿÿ@@@´
D__inference_conv2d_60_layer_call_and_return_conditional_losses_71002l=>7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
)__inference_conv2d_60_layer_call_fn_70991_=>7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@µ
D__inference_conv2d_61_layer_call_and_return_conditional_losses_71022mGH7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  @
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 
)__inference_conv2d_61_layer_call_fn_71011`GH7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  @
ª "!ÿÿÿÿÿÿÿÿÿ  ¶
D__inference_conv2d_62_layer_call_and_return_conditional_losses_71042nMN8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 
)__inference_conv2d_62_layer_call_fn_71031aMN8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª "!ÿÿÿÿÿÿÿÿÿ  ¶
D__inference_conv2d_63_layer_call_and_return_conditional_losses_71062nWX8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_63_layer_call_fn_71051aWX8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¶
D__inference_conv2d_64_layer_call_and_return_conditional_losses_71082n]^8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_64_layer_call_fn_71071a]^8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¶
D__inference_conv2d_65_layer_call_and_return_conditional_losses_71102ngh8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_65_layer_call_fn_71091agh8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¶
D__inference_conv2d_66_layer_call_and_return_conditional_losses_71122nmn8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_66_layer_call_fn_71111amn8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¶
D__inference_conv2d_67_layer_call_and_return_conditional_losses_71155n{|8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_67_layer_call_fn_71144a{|8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¸
D__inference_conv2d_68_layer_call_and_return_conditional_losses_71175p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_68_layer_call_fn_71164c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¸
D__inference_conv2d_69_layer_call_and_return_conditional_losses_71208p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 
)__inference_conv2d_69_layer_call_fn_71197c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª "!ÿÿÿÿÿÿÿÿÿ  ¸
D__inference_conv2d_70_layer_call_and_return_conditional_losses_71228p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 
)__inference_conv2d_70_layer_call_fn_71217c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª "!ÿÿÿÿÿÿÿÿÿ  ·
D__inference_conv2d_71_layer_call_and_return_conditional_losses_71261o£¤8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@À
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
)__inference_conv2d_71_layer_call_fn_71250b£¤8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@À
ª " ÿÿÿÿÿÿÿÿÿ@@@¶
D__inference_conv2d_72_layer_call_and_return_conditional_losses_71281n©ª7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 
)__inference_conv2d_72_layer_call_fn_71270a©ª7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@@@
ª " ÿÿÿÿÿÿÿÿÿ@@@º
D__inference_conv2d_73_layer_call_and_return_conditional_losses_71314r·¸9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ`
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_conv2d_73_layer_call_fn_71303e·¸9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ`
ª ""ÿÿÿÿÿÿÿÿÿ º
D__inference_conv2d_74_layer_call_and_return_conditional_losses_71334r½¾9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_conv2d_74_layer_call_fn_71323e½¾9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ º
D__inference_conv2d_75_layer_call_and_return_conditional_losses_71354rÃÄ9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_75_layer_call_fn_71343eÃÄ9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_68975R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_12_layer_call_fn_68981R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_68987R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_13_layer_call_fn_68993R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_68999R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_14_layer_call_fn_69005R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_69011R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_15_layer_call_fn_69017R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
B__inference_model_3_layer_call_and_return_conditional_losses_70230­6'(-.78=>GHMNWX]^ghmn{|£¤©ª·¸½¾ÃÄB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ô
B__inference_model_3_layer_call_and_return_conditional_losses_70341­6'(-.78=>GHMNWX]^ghmn{|£¤©ª·¸½¾ÃÄB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ó
B__inference_model_3_layer_call_and_return_conditional_losses_70757¬6'(-.78=>GHMNWX]^ghmn{|£¤©ª·¸½¾ÃÄA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ó
B__inference_model_3_layer_call_and_return_conditional_losses_70922¬6'(-.78=>GHMNWX]^ghmn{|£¤©ª·¸½¾ÃÄA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Ì
'__inference_model_3_layer_call_fn_69547 6'(-.78=>GHMNWX]^ghmn{|£¤©ª·¸½¾ÃÄB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿÌ
'__inference_model_3_layer_call_fn_70119 6'(-.78=>GHMNWX]^ghmn{|£¤©ª·¸½¾ÃÄB¢?
8¢5
+(
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿË
'__inference_model_3_layer_call_fn_705116'(-.78=>GHMNWX]^ghmn{|£¤©ª·¸½¾ÃÄA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿË
'__inference_model_3_layer_call_fn_705926'(-.78=>GHMNWX]^ghmn{|£¤©ª·¸½¾ÃÄA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿè
#__inference_signature_wrapper_70430À6'(-.78=>GHMNWX]^ghmn{|£¤©ª·¸½¾ÃÄE¢B
¢ 
;ª8
6
input_4+(
input_4ÿÿÿÿÿÿÿÿÿ"?ª<
:
	conv2d_75-*
	conv2d_75ÿÿÿÿÿÿÿÿÿî
K__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_69030R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_up_sampling2d_12_layer_call_fn_69036R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_69049R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_up_sampling2d_13_layer_call_fn_69055R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_69068R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_up_sampling2d_14_layer_call_fn_69074R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_up_sampling2d_15_layer_call_and_return_conditional_losses_69087R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_up_sampling2d_15_layer_call_fn_69093R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ