type SupportsJoin = {
    join(separator?: string): string;
}

export function toListLiteral(array: SupportsJoin) {
    return "[" + array.join(",") + "]";
}

export function toNpArrayLiteral(array: Uint8ClampedArray, shape: number[]) {
    return `np.array(${toListLiteral(array)}).reshape(${toListLiteral(shape)})`;
}
