export interface IPyProxy {
    length: number;
    type: string;
    
    copy(): IPyProxy;
    
    delete(key: any): void;
    
    destroy(destroyed_msg?: string): void;
    
    get(key: any): any;
    
    has(key: string): boolean;
    
    set(key: any, value: any): void;
}
