declare namespace tf {
    /**
     * Represents a multi-dimensional array of numbers used in TensorFlow.js computations.
     */
    class Tensor<T extends tf.DataType = 'float32'> {
        /** Shape of the tensor. */
        readonly shape: number[];
        /** Data type of the tensor. */
        readonly dtype: T;

        /**
         * Computes the difference between two slices along axis 1 (typically the 'border' dimension).
         * It slices at [startY:endY, startZ, :] and [startY:endY, endZ, :] and subtracts them.
         *
         * @param {number[]} yRange - A tuple [startY, endY] indicating the bscan slice range.
         * @param {number[]} zRange - A tuple [startZ, endZ] indicating the border slice indices.
         * @returns {Tensor<'int32'>} A tensor representing the difference between the two slices.
         */
        diffAlongAxis1(yRange: [number, number], zRange: [number, number]): tf.Tensor<'int32'>;

        /**
         * Replaces the current tensor's data with a new tensor, disposing of the old data if needed.
         * @param {Tensor} newTensor - The tensor whose data will replace the current tensor's data.
         * @returns {Tensor} - The updated tensor (this).
         */
        overwrite(newTensor: Tensor<T>): Tensor<T>;

        /**
         * Returns the tensor data as a nested array.
         */
        array(): Promise<any[]>;

        /**
         * Synchronously returns the tensor data as a nested array.
         */
        arraySync(): any[];

        /**
         * Returns a `TensorBuffer` that allows access to the tensor's values.
         */
        buffer(): Promise<TensorBuffer<any>>;

        /**
         * Synchronously returns a `TensorBuffer` that allows access to the tensor's values.
         */
        bufferSync(): TensorBuffer<any>;

        /**
         * Returns the tensor data as a typed array asynchronously.
         */
        data(): Promise<TypedArray>;

        /**
         * Synchronously returns the tensor data as a typed array.
         */
        dataSync(): TypedArray;

        /**
         * Disposes the tensor from memory.
         */
        dispose(): void;

        /**
         * Creates a copy of the tensor.
         */
        clone(): Tensor<T>;

        /**
         * Reshapes the tensor into the specified shape.
         */
        reshape(shape: number[]): Tensor<T>;

        /**
         * Expands the dimensions of the tensor at the specified axis.
         */
        expandDims(axis?: number): Tensor<T>;

        /**
         * Removes dimensions of size 1 from the shape of the tensor.
         */
        squeeze(axis?: number[]): Tensor<T>;

        /**
         * Adds another tensor to this tensor element-wise.
         */
        add(tensor: Tensor | number): Tensor<T>;

        /**
         * Subtracts another tensor from this tensor element-wise.
         */
        sub(tensor: Tensor | number): Tensor<T>;

        /**
         * Multiplies this tensor with another tensor element-wise.
         */
        mul(tensor: Tensor<T> | number): Tensor<T>;

        /**
         * Divides this tensor by another tensor element-wise.
         */
        div(tensor: Tensor<T> | number): Tensor<T>;

        /**
         * Computes the sum of tensor elements along specified axes.
         * @param axis Axes to sum over.
         * @param keepDims If true, retains reduced dimensions with size 1.
         */
        sum(axis?: number | number[], keepDims?: boolean): Tensor<T>;

        /**
         * Computes the mean of tensor elements along specified axes.
         * @param axis Axes to compute the mean over.
         * @param keepDims If true, retains reduced dimensions with size 1.
         */
        mean(axis?: number | number[], keepDims?: boolean): Tensor<T>;

        /**
         * Casts the tensor to a different data type.
         * @param dtype The desired data type.
         */
        cast<D extends tf.DataType>(dtype: D): Tensor<D>;

        /**
         * Extracts a slice from a tf.Tensor starting at coordinates *begin* and is of size *size*.
         * @param begin The coordinates to start the slice from.
         * @param size The size of the slice.
         */
        slice(begin: number | number[], size?: number | number[]): Tensor<T>;

        transpose(perm?: number[] | null): Tensor<T>;

        /**
         * Computes the minimum value in this tensor.
         * If an axis is specified, computes along that axis.
         * @param axis The axis along which to compute the minimum. If not provided, computes over all elements.
         * @param keepDims If true, retains reduced dimensions with size 1. Default: false.
         * @returns A tensor with the minimum value(s).
         */
        min(axis?: number | number[], keepDims?: boolean): Tensor<T>;

        /**
         * Computes the maximum value in this tensor.
         * If an axis is specified, computes along that axis.
         * @param axis The axis along which to compute the maximum. If not provided, computes over all elements.
         * @param keepDims If true, retains reduced dimensions with size 1. Default: false.
         * @returns A tensor with the maximum value(s).
         */
        max(axis?: number | number[], keepDims?: boolean): Tensor<T>;

        /**
         * Returns a tensor with the negative of each element.
         * @returns {Tensor}
         */
        neg(): Tensor<T>;

        /**
         * Computes the absolute value of each element.
         * @returns {Tensor}
         */
        abs(): Tensor<T>;

        /**
         * Computes the square root of each element.
         * @returns {Tensor}
         */
        sqrt(): Tensor<T>;

        /**
         * Computes the square of each element.
         * @returns {Tensor}
         */
        square(): Tensor<T>;

        /**
         * Computes the natural logarithm of each element.
         * @returns {Tensor}
         */
        log(): Tensor<T>;

        /**
         * Computes the exponential of each element.
         * @returns {Tensor}
         */
        exp(): Tensor<T>;

        /**
         * Returns a tensor with boolean values where elements are not equal to zero.
         * @returns {Tensor}
         */
        notEqual(tensor: Tensor): Tensor<T>;

        /**
         * Returns a tensor with boolean values where elements are equal to the corresponding elements in another tensor.
         * @param {Tensor} tensor - Tensor to compare with.
         * @returns {Tensor}
         */
        equal(tensor: Tensor): Tensor<T>;

        /**
         * Computes the logical OR across specified axes.
         * @param {number|number[]} [axis] - The dimensions to reduce.
         * @param {boolean} [keepDims=false] - If true, retains reduced dimensions with length 1.
         * @returns {Tensor}
         */
        any(axis?: number | number[], keepDims?: boolean): Tensor<T>;

        /**
         * Computes the logical AND across specified axes.
         * @param {number|number[]} [axis] - The dimensions to reduce.
         * @param {boolean} [keepDims=false] - If true, retains reduced dimensions with length 1.
         * @returns {Tensor}
         */
        all(axis?: number | number[], keepDims?: boolean): Tensor<T>;

        /**
         * Gathers slices from this tensor along an axis specified by `axis`.
         * @param {Tensor|number[]} indices - The indices of elements to gather.
         * @param {number} [axis=0] - The axis along which to gather.
         * @returns {Tensor}
         */
        gather(indices: Tensor<'int32'> | number[], axis?: number): Tensor<T>;

        /**
         * Repeats the tensor along each dimension.
         * @param reps An array of numbers specifying the number of times to replicate the tensor along each axis.
         * @returns A tiled tensor.
         */
        tile(reps: number[]): Tensor<T>;

        /**
         * Casts the tensor to float32 data type.
         * @returns {tf.Tensor<'float32'>} A new tensor with float32 dtype.
         */
        toFloat(): tf.Tensor<'float32'>;

        /**
         * Returns the element-wise comparison of the tensor and a scalar/tensor, checking if elements are greater than or equal to the provided value.
         * @param {number | tf.Tensor} x The scalar or tensor to compare with.
         * @returns {tf.Tensor<'bool'>} A boolean tensor indicating where the condition holds.
         */
        greaterEqual(x: number | tf.Tensor): tf.Tensor<'bool'>;

        /**
         * Returns the element-wise logical AND operation with another tensor.
         * @param {tf.Tensor<'bool'>} x The tensor to perform logical AND with.
         * @returns {tf.Tensor<'bool'>} A boolean tensor indicating where both tensors are true.
         */
        logicalAnd(x: tf.Tensor<'bool'>): tf.Tensor<'bool'>;

        /**
         * Returns the element-wise comparison of the tensor and a scalar/tensor, checking if elements are less than or equal to the provided value.
         * @param {number | tf.Tensor} x The scalar or tensor to compare with.
         * @returns {tf.Tensor<'bool'>} A boolean tensor indicating where the condition holds.
         */
        lessEqual(x: number | tf.Tensor): tf.Tensor<'bool'>;

        /**
         * Returns a tensor with each element rounded to the nearest integer.
         * @returns A tensor with rounded values.
         */
        round(): this;

        /**
         * Scale the tensor to values of [0,1] with minmax
         */
        minmax(): Tensor<T>;

        /**
         * Expand grayscale values to RGBA (R=G=B, A=255).
         */
        toRGBA(): Uint8ClampedArray;

        /**
         * Expand grayscale values to RGBA (R=G=B, A=255).
         */
        draw(): HTMLCanvasElement;
    }

    class TensorBuffer<T extends DataType = 'float32'> {
        readonly shape: number[];
        readonly dtype: T;
        values: TypedArray;

        get(...locs: number[]): number;
        set(value: number, ...locs: number[]): void;

        toTensor(): Tensor<T>;
    }

    function buffer<T extends DataType>(
        shape: number[],
        dtype?: T,
        values?: TypedArray,
    ): TensorBuffer<T>;

    /**
     * Data types supported by TensorFlow.js tensors.
     */
    type DataType = 'float32' | 'int32' | 'bool' | 'complex64';

    /**
     * Typed arrays supported by TensorFlow.js.
     */
    type TypedArray = Float32Array | Int32Array | Uint8Array;

    /**
     * Creates a tensor with the specified values, shape, and data type.
     * @param values The values to store in the tensor.
     * @param shape The shape of the tensor.
     * @param dtype The data type of the tensor.
     */
    function tensor<T extends DataType = 'float32'>(
        values: TypedArray | number[] | number[][],
        shape?: number[],
        dtype?: T,
    ): Tensor<T>;

    /**
     * Creates a 1-dimensional tensor with the specified values and data type.
     * @param values The values to store in the tensor.
     * @param dtype The data type of the tensor.
     */
    function tensor1d<T extends DataType = 'float32'>(
        values: TypedArray | number[],
        dtype?: T,
    ): Tensor<T>;

    /**
     * Creates a 2-dimensional tensor with the specified values, shape, and data type.
     * @param values The values to store in the tensor.
     * @param shape The shape of the tensor.
     * @param dtype The data type of the tensor.
     */
    function tensor2d<T extends DataType = 'float32'>(
        values: TypedArray | number[][],
        shape?: [number, number],
        dtype?: T,
    ): Tensor<T>;

    /**
     * Creates a 3-dimensional tensor with the specified values, shape, and data type.
     * @param values The values to store in the tensor.
     * @param shape The shape of the tensor.
     * @param dtype The data type of the tensor.
     */
    function tensor3d<T extends DataType = 'float32'>(
        values: TypedArray | number[][][],
        shape?: [number, number, number],
        dtype?: T,
    ): Tensor<T>;

    /**
     * Creates a 4-dimensional tensor with the specified values, shape, and data type.
     * @param values The values to store in the tensor.
     * @param shape The shape of the tensor.
     * @param dtype The data type of the tensor.
     */
    function tensor4d<T extends DataType = 'float32'>(
        values: TypedArray | number[][][][],
        shape?: [number, number, number, number],
        dtype?: T,
    ): Tensor<T>;

    /**
     * Adds two tensors element-wise.
     * @param a The first tensor.
     * @param b The second tensor.
     */
    function add<T extends DataType>(a: Tensor<T>, b: Tensor<T> | number): Tensor<T>;

    /**
     * Subtracts the second tensor from the first tensor element-wise.
     * @param a The first tensor.
     * @param b The second tensor.
     */
    function sub<T extends DataType>(a: Tensor<T>, b: Tensor<T> | number): Tensor<T>;

    /**
     * Multiplies two tensors element-wise.
     * @param a The first tensor.
     * @param b The second tensor.
     */
    function mul<T extends DataType>(a: Tensor<T>, b: Tensor<T> | number): Tensor<T>;

    /**
     * Divides the first tensor by the second tensor element-wise.
     * @param a The first tensor.
     * @param b The second tensor.
     */
    function div<T extends DataType>(a: Tensor<T>, b: Tensor<T> | number): Tensor<T>;

    /**
     * Computes the sum of elements across a tensor.
     * @param x The input tensor.
     * @param axis The axes to sum over.
     * @param keepDims If true, retains reduced dimensions with size 1.
     */
    function sum<T extends DataType>(
        x: Tensor<T>,
        axis?: number | number[],
        keepDims?: boolean,
    ): Tensor<T>;

    /**
     * Disposes a tensor or a list of tensors from memory.
     * @param tensor The tensor or tensors to dispose.
     */
    function dispose(tensor: Tensor | Tensor[]): void;

    /**
     * Executes a function within a clean memory scope, disposing intermediate tensors.
     * @param nameOrFn The name or the function to execute.
     * @param fn The function to execute (if `nameOrFn` is a string).
     */
    function tidy<T>(nameOrFn: string | (() => T), fn?: () => T): T;

    /**
     * The version of TensorFlow.js core.
     */
    const version_core: string;

    /**
     * Computes the minimum value of a tensor along an axis.
     * @param x The input tensor.
     * @param axis The axis along which to compute the minimum. If not provided, computes over all elements.
     * @param keepDims If true, retains reduced dimensions with size 1. Default: false.
     * @returns A tensor with the minimum value(s).
     */
    function min<T extends Tensor>(x: T, axis?: number | number[], keepDims?: boolean): Tensor;

    /**
     * Computes the maximum value of a tensor along an axis.
     * @param x The input tensor.
     * @param axis The axis along which to compute the maximum. If not provided, computes over all elements.
     * @param keepDims If true, retains reduced dimensions with size 1. Default: false.
     * @returns A tensor with the maximum value(s).
     */
    function max<T extends Tensor>(x: T, axis?: number | number[], keepDims?: boolean): Tensor;

    /**
     * Returns a tensor with boolean values where elements of `a` and `b` are not equal.
     * @param {Tensor} a - First tensor to compare.
     * @param {Tensor} b - Second tensor to compare.
     * @returns {Tensor}
     */
    function notEqual<T extends DataType>(a: Tensor<T>, b: Tensor<T>): Tensor<T>;

    /**
     * Concatenates a list of tensors along a given axis.
     * @param {Tensor[]} tensors - Array of tensors to concatenate.
     * @param {number} [axis=0] - Axis along which to concatenate.
     * @returns {Tensor}
     */
    function concat<T extends DataType>(tensors: Tensor<T>[], axis?: number): Tensor<T>;

    /**
     * Computes the mean of elements across dimensions of a tensor.
     * @param {Tensor} tensor - The tensor to reduce.
     * @param {number|number[]} [axis] - The dimensions to reduce.
     * @param {boolean} [keepDims=false] - If true, retains reduced dimensions with length 1.
     * @returns {Tensor}
     */
    function mean<T extends DataType>(
        tensor: Tensor<T>,
        axis?: number | number[],
        keepDims?: boolean,
    ): Tensor<T>;

    /**
     * Transposes the dimensions of a tensor.
     * @param {Tensor} tensor - The tensor to transpose.
     * @param {number[]} [perm] - The permutation of the dimensions.
     * @returns {Tensor}
     */
    function transpose<T extends DataType>(tensor: Tensor<T>, perm?: number[]): Tensor<T>;

    /**
     * Extracts a slice from a tensor.
     * @param {Tensor} tensor - The tensor to slice.
     * @param {number[]} begin - Coordinates to start slicing.
     * @param {number[]} size - Size of the slice.
     * @returns {Tensor}
     */
    function slice<T extends DataType>(
        tensor: Tensor<T>,
        begin: number[],
        size: number[],
    ): Tensor<T>;

    /**
     * Returns a tensor with boolean values where elements of `a` and `b` are equal.
     * @param {Tensor} a - First tensor to compare.
     * @param {Tensor} b - Second tensor to compare.
     * @returns {Tensor}
     */
    function equal<T extends DataType>(a: Tensor<T>, b: Tensor<T>): Tensor<T>;

    /**
     * Computes the logical OR across specified axes.
     * @param {Tensor} tensor - The tensor to reduce.
     * @param {number|number[]} [axis] - The dimensions to reduce.
     * @param {boolean} [keepDims=false] - If true, retains reduced dimensions with length 1.
     * @returns {Tensor}
     */
    function any<T extends DataType>(
        tensor: Tensor<T>,
        axis?: number | number[],
        keepDims?: boolean,
    ): Tensor<T>;

    /**
     * Computes the logical AND across specified axes.
     * @param {Tensor} tensor - The tensor to reduce.
     * @param {number|number[]} [axis] - The dimensions to reduce.
     * @param {boolean} [keepDims=false] - If true, retains reduced dimensions with length 1.
     * @returns {Tensor}
     */
    function all<T extends DataType>(
        tensor: Tensor<T>,
        axis?: number | number[],
        keepDims?: boolean,
    ): Tensor<T>;

    /**
     * Gathers slices from `tensor` along an axis specified by `axis`.
     * @param {Tensor} tensor - The source tensor.
     * @param {Tensor|number[]} indices - The indices of elements to gather.
     * @param {number} [axis=0] - The axis along which to gather.
     * @returns {Tensor}
     */
    function gather<T extends DataType>(
        tensor: Tensor<T>,
        indices: Tensor | number[],
        axis?: number,
    ): Tensor<T>;

    /**
     * Creates a 1-D tensor containing a sequence of numbers.
     * @param start The start of the range (inclusive).
     * @param stop The end of the range (exclusive).
     * @param step The difference between consecutive values. Defaults to 1.
     * @param dtype The data type of the result tensor. Defaults to 'float32'.
     * @returns A 1-D tensor with values from start to stop.
     */
    function range(start: number, stop: number, step?: number, dtype?: DataType): Tensor;

    /**
     * Constructs a tensor by repeating the input tensor along each dimension.
     * @param tensor The tensor to tile.
     * @param reps An array of numbers specifying the number of times to replicate the tensor along each axis.
     * @returns A tiled tensor.
     */
    function tile<T extends DataType>(tensor: Tensor<T>, reps: number[]): Tensor<T>;

    /**
     * Stacks a list of rank-R tensors into one rank-(R+1) tensor.
     * @param tensors An array of tensors with the same shape.
     * @param axis The axis along which to stack. Defaults to 0.
     * @returns A stacked tensor.
     */
    function stack<T extends DataType>(tensors: Array<Tensor<T>>, axis?: number): Tensor<T>;

    /**
     * Gathers slices from params into a Tensor with shape specified by indices.
     * @param params The tensor from which to gather values.
     * @param indices Index tensor indicating which values to gather.
     * @returns A tensor with gathered values.
     */
    function gatherND<T extends DataType>(
        params: Tensor<T>,
        indices: Tensor<'int32'> | number[][],
    ): Tensor<T>;

    /**
     * Creates a tensor filled with zeros, matching the shape and type of the provided tensor.
     * @param {tf.Tensor} x The tensor to mimic.
     * @returns {tf.Tensor} A tensor with the same shape and dtype as x, filled with zeros.
     */
    function zerosLike<T extends tf.DataType>(x: tf.Tensor<T>): tf.Tensor<T>;

    /**
     * Selects elements from either the true or false tensor based on the condition tensor.
     * @param {tf.Tensor<'bool'>} condition A boolean tensor where true indicates selection from `a`.
     * @param {tf.Tensor} a Tensor to select elements from when condition is true.
     * @param {tf.Tensor} b Tensor to select elements from when condition is false.
     * @returns {tf.Tensor} A tensor with elements chosen from `a` or `b` based on `condition`.
     */
    function where<T extends tf.DataType>(
        condition: tf.Tensor<'bool'>,
        a: tf.Tensor<T>,
        b: tf.Tensor<T>,
    ): tf.Tensor<T>;

    /**
     * Returns the element-wise comparison of two tensors, checking if elements in the first tensor are greater than or equal to elements in the second tensor.
     * @param {tf.Tensor} a The first tensor.
     * @param {tf.Tensor} b The second tensor.
     * @returns {tf.Tensor<'bool'>} A boolean tensor indicating where the condition holds.
     */
    function greaterEqual<T extends tf.DataType>(
        a: tf.Tensor<T>,
        b: tf.Tensor<T>,
    ): tf.Tensor<'bool'>;

    /**
     * Performs element-wise logical AND operation on two tensors.
     * @param {tf.Tensor<'bool'>} a The first boolean tensor.
     * @param {tf.Tensor<'bool'>} b The second boolean tensor.
     * @returns {tf.Tensor<'bool'>} A boolean tensor indicating where both tensors are true.
     */
    function logicalAnd(a: tf.Tensor<'bool'>, b: tf.Tensor<'bool'>): tf.Tensor<'bool'>;

    /**
     * Returns the element-wise comparison of two tensors, checking if elements in the first tensor are less than or equal to elements in the second tensor.
     * @param {tf.Tensor} a The first tensor.
     * @param {tf.Tensor} b The second tensor.
     * @returns {tf.Tensor<'bool'>} A boolean tensor indicating where the condition holds.
     */
    function lessEqual<T extends tf.DataType>(a: tf.Tensor<T>, b: tf.Tensor<T>): tf.Tensor<'bool'>;

    /**
     * Creates a tensor filled with ones.
     *
     * @param shape - The shape of the output tensor.
     * @param dtype - The data type of the tensor. Defaults to 'float32'.
     * @returns A tensor of specified shape filled with ones.
     */
    function ones<T extends tf.Tensor>(shape: number | number[], dtype?: tf.DataType): T;

    /**
     * Creates a tensor filled with zeros.
     *
     * @param shape - The shape of the output tensor.
     * @param dtype - The data type of the tensor. Defaults to 'float32'.
     * @returns A tensor of specified shape filled with zeros.
     */
    function zeros<T extends tf.Tensor>(shape: number | number[], dtype?: tf.DataType): T;

    /**
     * Rounds each element of the tensor to the nearest integer.
     * @param x The input tensor.
     * @returns A tensor with rounded values.
     */
    function round<T extends Tensor>(x: T): T;
}
