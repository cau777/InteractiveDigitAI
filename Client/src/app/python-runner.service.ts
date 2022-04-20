import {Inject, Injectable} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {firstValueFrom} from "rxjs";
import {DOCUMENT} from "@angular/common";
import {
    IPyodideInitMessage,
    IPyodideRunExpressionMessage,
    IPyodideSelectMessage, PyodideWorkerMessage
} from "./pyodide-worker-messages";

type ScriptName = "test";

@Injectable({
    providedIn: 'root'
})
export class PythonRunnerService {
    private worker?: Worker;
    
    public constructor(private httpClient: HttpClient,
                       @Inject(DOCUMENT) private document: Document) {
    }
    
    public async loadScript(name: ScriptName) {
        await this.initWorker();
        
        let code = await this.getCode(name);
        let dependencies = await this.getDependencies(name);
        let message: IPyodideSelectMessage = {action: "select", code: code, data: dependencies};
        await this.waitWorker(message);
    }
    
    public async run(code: string) {
        let message: IPyodideRunExpressionMessage = {action: "run", expression: code};
        
        return await new Promise<any>((resolve, reject) => {
            this.worker!.postMessage(message);
            this.worker!.addEventListener("error", e => {
                // TODO: redirect
                reject(e.error);
            });
            this.worker!.addEventListener("message", e => {
                let data: PyodideWorkerMessage = e.data;
                
                if (data.action === "output") {
                    console.log(data.content); // TODO: redirect
                } else if (data.action === "result") {
                    resolve(data.content);
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
        return [];
    }
}
