import {PyProxy} from "./pyproxy";
import {IConsoleFuture} from "./iconsole-future";

export type PyConsole = {
    stdout_callback: (message: string) => void;
    stderr_callback: (message: string) => void;
    globals: any;
    
    push(line: string): IConsoleFuture;
} | PyProxy;
