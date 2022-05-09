// type SupportsJoin = {
//     join(separator?: string): string;
// }
//
// export function toListLiteral(array: SupportsJoin) {
//     return "[" + array.join(",") + "]";
// }
//
// export function toNpArrayLiteral(array: SupportsJoin, shape: number[]) {
//     return `np.array(${toListLiteral(array)}).reshape(${toListLiteral(shape)})`;
// }
