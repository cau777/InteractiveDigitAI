import {ObjDict} from "../utils";

export type PythonRunCallback = (content: string, isError: boolean) => void;

export interface IPyodideRunner {
    load(libs: string[], libsArchives: ArrayBuffer[], output?: PythonRunCallback): Promise<void>;
    
    select(code: string, params: ObjDict<any>): Promise<void>;
    
    run(expression: string, params: ObjDict<any>, callback?: PythonRunCallback): Promise<any>;
}
