import {PyProxy} from "./pyproxy";

export type PyConsole = {
    stdout_callback: (message: string) => void;
    stderr_callback: (message: string) => void;
    globals: any;
    
    push(line: string): IConsoleFuture;
} & PyProxy;

export interface IConsoleFuture extends Promise<any> {
    syntax_check: "incomplete" | "complete" | "syntax-error";
    formatted_error: string;
}
