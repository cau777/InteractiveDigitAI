export interface IPyodideInitMessage {
    action: "init";
    baseUrl: string;
}

export interface IPyodideSelectMessage {
    action: "select";
    code: string;
    data: any[];
}

export interface IPyodideCloseMessage {
    action: "close";
}

export interface IPyodideRunExpressionMessage {
    action: "run";
    expression: string;
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
