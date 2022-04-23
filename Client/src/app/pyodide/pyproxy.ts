export type PyProxy = {
    length: number;
    type: string;
    
    copy(): PyProxy;
    
    delete(key: any): void;
    
    destroy(destroyed_msg?: string): void;
    
    get(key: any): any;
    
    has(key: string): boolean;
    
    set(key: any, value: any): void;
    
    toJs(options?: { 
        create_pyproxies?: boolean, depth?: number, pyproxies?: PyProxy[], 
        default_converter?: (obj: PyProxy, convert: {}, cacheConversion: {}) => any, 
        dict_converter?: (array: Iterable<[key: string, value: any]>) => any }): any;
} | any;
