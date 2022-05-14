import {Injectable} from '@angular/core';
import {firstValueFrom} from "rxjs";
import {PyodideInitMessage, PyodideRunMessage, PyodideSelectMessage, PyodideWorkerMessage} from "./pyodide-worker-messages";
import {AiName} from "./ai-repos.service";
import {ObjDict} from "./utils";
import {AssetsHttpClientService} from "./assets-http-client.service";

export type ScriptName = "test" | AiName;
export type PythonRunCallback = (content: string, isError: boolean) => void;

// TODO: fix error when worker is busy

@Injectable({
    providedIn: 'root'
})
export class PythonRunnerService {
    private readonly worker: Promise<Worker>;
    private loaded?: ScriptName;
    
    public constructor(private assetsClient: AssetsHttpClientService) {
        this.worker = this.initWorker();
    }
    
    public async loadScript(name: ScriptName, params: ObjDict<any> = {}) {
        if (name === this.loaded) return;
        let worker = await this.worker;
        
        let code = await this.getCode(name);
        let message: PyodideSelectMessage = {action: "select", code: code, params: params};
        await this.waitWorker(worker, message);
        
        console.log(name + " loaded");
        this.loaded = name;
    }
    
    public async run(code: string, params: ObjDict<any> = {}, output?: PythonRunCallback) {
        let message: PyodideRunMessage = {
            action: "run",
            expression: code,
            params: params
        };
        let worker = await this.worker;
        
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
        if (typeof Worker === "undefined") throw TypeError("Workers are not supported in your system") // TODO: add support without workers
        let worker = new Worker(new URL("./pyodide.worker", import.meta.url));
        
        let archives = await this.loadLibsArchives("python/codebase-0.0.1-py3-none-any");
        let libs = ["numpy"];
        let init: PyodideInitMessage = {action: "init", libsArchives: archives, libs: libs};
        await this.waitWorker(worker, init);
        
        return worker;
    }
    
    private async loadLibsArchives(...names: string[]) {
        let libs = [];
        for (let name of names) {
            libs.push(await firstValueFrom(this.assetsClient.get(name + ".whl", {responseType: "arraybuffer"})));
        }
        return libs;
    }
    
    private async waitWorker(worker: Worker, message: any) {
        return new Promise((resolve, reject) => {
            worker.postMessage(message);
            worker.addEventListener("error", reject);
            worker.addEventListener("message", e => resolve(e.data));
        });
    }
    
    private async getCode(name: ScriptName): Promise<string> {
        return firstValueFrom(this.assetsClient.get(`python/${name}.py`, {responseType: "text"}));
    }
}
