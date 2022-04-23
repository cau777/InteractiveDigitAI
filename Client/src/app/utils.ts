export function zip<T1, T2>(arr1: T1[], arr2: T2[]): [T1, T2][] {
    if (arr1.length !== arr2.length) throw RangeError("Arrays must have the same length: " + arr1.length + ":" + arr2.length);
    const result: [T1, T2][] = [];
    
    for (let i = 0; i < arr1.length; i++) {
        result[i] = [arr1[i], arr2[i]];
    }
    
    return result;
}

export function arrayBufferToString(buffer: ArrayBuffer) {
    let result = "";
    let array = new Uint8Array(buffer);
    for (let code of array) {
        result += String.fromCharCode(code);
    }
    return result;
}
