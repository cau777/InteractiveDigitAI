import {IPyodideRunner, PythonRunCallback} from "./IPyodideRunner";
import {ObjDict} from "../utils";
import {
    PyodideLoadMessage,
    PyodideRunMessage,
    PyodideSelectMessage,
    PyodideWorkerMessage
} from "../pyodide-worker-messages";

export class PyodideWorkerInterface implements IPyodideRunner {
    private worker!: Worker;
    
    public async load(libs: string[], libsArchives: ArrayBuffer[], callback: PythonRunCallback) {
        let worker = new Worker(new URL("../pyodide.worker", import.meta.url));
        let init: PyodideLoadMessage = {action: "load", libsArchives: libsArchives, libs: libs};
        
        await this.waitWorker(worker, init, callback);
        
        this.worker = worker;
    }
    
    public async select(code: string, params: ObjDict<any>): Promise<void> {
        let message: PyodideSelectMessage = {
            action: "select", code: code,
            params: params
        };
        let worker = await this.worker;
        await this.waitWorker(worker, message)
    }
    
    public async run(expression: string, params: ObjDict<any>, callback: PythonRunCallback): Promise<any> {
        let message: PyodideRunMessage = {
            action: "run",
            expression: expression,
            params: params
        };
        let worker = await this.worker;
        return await this.waitWorker(worker, message, callback);
    }
    
    private async waitWorker(worker: Worker, message: any, callback?: PythonRunCallback) {
        return new Promise<any>((resolve, reject) => {
            worker.postMessage(message);
            
            worker.addEventListener("error", (e) => {
                console.error(e);
                callback?.(e.message, true);
                reject(e);
            });
            
            worker.addEventListener("message", e => {
                let data: PyodideWorkerMessage = e.data;
                
                if (data.action === "output") {
                    callback?.(data.content, data.isError);
                } else if (data.action === "result") {
                    resolve(data.content);
                    worker.removeAllListeners?.();
                }
            });
        });
    }
}
