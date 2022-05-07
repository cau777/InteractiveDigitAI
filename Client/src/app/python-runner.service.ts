import {Inject, Injectable} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {firstValueFrom, zip} from "rxjs";
import {DOCUMENT} from "@angular/common";
import {
    IPyodideInitMessage,
    IPyodideRunExpressionMessage,
    IPyodideSelectMessage,
    PyodideWorkerMessage
} from "./pyodide-worker-messages";
import {AiName, AiReposService} from "./ai-repos.service";
import {arrayBufferToString} from "./utils";

export type ScriptName = "test" | AiName;
export type PythonRunCallback = (content: string, isError: boolean) => void;

@Injectable({
    providedIn: 'root'
})
export class PythonRunnerService {
    private worker?: Worker;
    private loaded?: ScriptName;
    
    public constructor(private httpClient: HttpClient,
                       private aiRepos: AiReposService,
                       @Inject(DOCUMENT) private document: Document) {
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
    
    public async run(code: string, output?: PythonRunCallback) {
        let message: IPyodideRunExpressionMessage = {action: "run", expression: code};
        let worker = this.worker!;
        
        return await new Promise<any>((resolve, reject) => {
            worker.postMessage(message);
            
            worker.addEventListener("error", e => {
                console.log(e);
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
            let init: IPyodideInitMessage = {action: "init", baseUrl: this.document.location.origin}
            return this.waitWorker(init);
        }
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
                let aiJson = await this.aiRepos.readAiJson("digit_recognition");
                result.set("model_data", aiJson);
                
                let [train, test] = await firstValueFrom(zip<readonly ArrayBuffer[]>([
                        this.httpClient.get("assets/mnist_test.dat", { // TODO: switch to train set
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
