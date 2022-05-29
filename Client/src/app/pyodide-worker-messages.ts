import {ObjDict} from "./utils";

export type PyodideLoadMessage = {
    action: "load";
    libsArchives: ArrayBuffer[];
    libs: string[];
}

export type PyodideSelectMessage = {
    action: "select";
    code: string;
    params: ObjDict<any>;
}

export type PyodideCloseMessage = {
    action: "close";
}

export type PyodideRunMessage = {
    action: "run";
    expression: string;
    params: ObjDict<any>;
}

export type PyodideOutputMessage = {
    action: "output";
    content: string;
    isError: boolean;
}

export type PyodideResultMessage = {
    action: "result";
    content?: any;
}

export type PyodideWorkerMessage = PyodideSelectMessage | PyodideRunMessage | PyodideOutputMessage |
    PyodideLoadMessage | PyodideResultMessage | PyodideCloseMessage;
