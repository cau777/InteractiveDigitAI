import {IPyProxy} from "./ipyproxy";
import {IConsoleFuture} from "./iconsole-future";

export interface IPyConsole extends IPyProxy {
    stdout_callback: (message: string) => void;
    stderr_callback: (message: string) => void;
    globals: any;
    
    push(line: string): IConsoleFuture;
}
