import {ObjDict} from "./utils";

export interface IPyodideInitMessage {
    action: "init";
    libs: ArrayBuffer[];
}

export interface IPyodideSelectMessage {
    action: "select";
    code: string;
    data: ObjDict<any>;
}

export interface IPyodideCloseMessage {
    action: "close";
}

export interface IPyodideRunExpressionMessage {
    action: "run";
    expression: string;
    params: ObjDict<any>;
}

export interface IPyodideOutputMessage {
    action: "output";
    content: string;
    isError: boolean;
}

export interface IPyodideResultMessage {
    action: "result";
    content?: any;
}

export type PyodideWorkerMessage = IPyodideSelectMessage | IPyodideRunExpressionMessage | IPyodideOutputMessage |
    IPyodideInitMessage | IPyodideResultMessage | IPyodideCloseMessage;
