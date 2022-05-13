import {Injectable} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {firstValueFrom, zip} from "rxjs";
import {
    IPyodideInitMessage,
    IPyodideRunExpressionMessage,
    IPyodideSelectMessage,
    PyodideWorkerMessage
} from "./pyodide-worker-messages";
import {AiName} from "./ai-repos.service";
import {arrayBufferToString, ObjDict} from "./utils";

export type ScriptName = "test" | AiName;
export type PythonRunCallback = (content: string, isError: boolean) => void;

@Injectable({
    providedIn: 'root'
})
export class PythonRunnerService {
    private worker?: Worker;
    private loaded?: ScriptName;
    
    public constructor(private httpClient: HttpClient) {
    }
    
    public async loadScript(name: ScriptName) {
        if (name === this.loaded) return;
        await this.initWorker();
        
        let code = await this.getCode(name);
        let dependencies = await this.getDependencies(name);
        let message: IPyodideSelectMessage = {action: "select", code: code, data: dependencies};
        await this.waitWorker(message);
        
        console.log(name + " loaded");
        this.loaded = name;
    }
    
    public async run(code: string, params: ObjDict<any> = {}, output?: PythonRunCallback) {
        let message: IPyodideRunExpressionMessage = {action: "run", expression: code, params: params};
        let worker = this.worker!;
        
        return await new Promise<any>((resolve, reject) => {
            worker.postMessage(message);
            
            worker.addEventListener("error", e => {
                console.error(e);
                output?.(e.message, true);
                reject(e.error);
            });
            
            worker.addEventListener("message", e => {
                let data: PyodideWorkerMessage = e.data;
                
                if (data.action === "output") {
                    console.log(data.content); // TODO: remove
                    output?.(data.content, data.isError);
                } else if (data.action === "result") {
                    resolve(data.content);
                    worker.removeAllListeners?.();
                }
            });
            
        });
    }
    
    private async initWorker() {
        if (this.worker === undefined) {
            if (typeof Worker === "undefined") throw TypeError("Workers are not supported in your system") // TODO: add support without workers
            this.worker = new Worker(new URL("./pyodide.worker", import.meta.url));
            
            let libs = await this.loadLibs("python/codebase-0.0.1-py3-none-any");
            let init: IPyodideInitMessage = {action: "init", libs: libs}
            return this.waitWorker(init);
        }
    }
    
    private async loadLibs(...names: string[]) {
        let libs = [];
        for (let name of names) {
            libs.push(await firstValueFrom(this.httpClient.get("/assets/" + name + ".whl", {responseType: "arraybuffer"})));
        }
        return libs;
    }
    
    private async waitWorker(message: any) {
        return new Promise((resolve, reject) => {
            this.worker!.postMessage(message);
            this.worker!.addEventListener("error", reject);
            this.worker!.addEventListener("message", e => resolve(e.data));
        });
    }
    
    private async getCode(name: ScriptName): Promise<string> {
        let observable = this.httpClient.get(`assets/python/${name}.py`, {responseType: "text"});
        return await firstValueFrom(observable);
    }
    
    private async getDependencies(name: ScriptName) {
        let result = new Map<string, any>();
        
        switch (name) {
            case "test":
                break;
            case "digit_recognition":
                let [train, test] = await firstValueFrom(zip<readonly ArrayBuffer[]>([
                        this.httpClient.get("assets/mnist_train.dat", {
                            responseType: "arraybuffer"
                        }),
                        this.httpClient.get("assets/mnist_test.dat", {
                            responseType: "arraybuffer"
                        })
                    ])
                );
                result.set("train_data", arrayBufferToString(train));
                result.set("test_data", arrayBufferToString(test));
                
                break;
        }
        
        return result;
    }
}
