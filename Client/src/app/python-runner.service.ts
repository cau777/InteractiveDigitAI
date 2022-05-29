import {Inject, Injectable} from '@angular/core';
import {firstValueFrom} from "rxjs";
import {AiName} from "./ai-repos.service";
import {ObjDict} from "./utils";
import {AssetsHttpClientService} from "./assets-http-client.service";
import {IPyodideRunner, PythonRunCallback} from "./pyodide/IPyodideRunner";
import {PyodideWorkerInterface} from "./pyodide/PyodideWorkerInterface";
import {DOCUMENT} from "@angular/common";

export type ScriptName = "test" | AiName;

const loadingPrefix = "Loading Pyodide";

// TODO: fix error when worker is busy

@Injectable({
    providedIn: 'root'
})
export class PythonRunnerService {
    public loadStatus?: string = loadingPrefix;
    private readonly pyodideInterface: Promise<IPyodideRunner>;
    private loaded?: ScriptName;
    private window: Window;
    
    public constructor(private assetsClient: AssetsHttpClientService,
                       @Inject(DOCUMENT) document: Document) {
        this.pyodideInterface = this.prepare();
        this.window = document.defaultView as Window;
    }
    
    public async loadScript(name: ScriptName, params: ObjDict<any> = {}) {
        if (name === this.loaded) return;
        let code = await this.getCode(name);
        
        await (await this.pyodideInterface).select(code, params);
        
        console.log(name + " loaded");
        this.loaded = name;
    }
    
    public async run(code: string, params: ObjDict<any> = {}, output?: PythonRunCallback) {
        return (await this.pyodideInterface).run(code, params, output);
    }
    
    private async prepare() {
        let archives = await this.loadLibsArchives("python/codebase-0.0.1-py3-none-any");
        let libs = ["numpy"];
        let callback: PythonRunCallback = (content, isError) => {
            if (isError) this.window.alert(content);
            else this.loadStatus = loadingPrefix + ": " + content;
        }
        
        // Adding support without web workers is not necessary because almost all browsers that support wasm also support worker
        if (typeof Worker === "undefined") throw new Error("Support for web workers is required");
        
        let pyodideInterface = new PyodideWorkerInterface();
        
        await pyodideInterface.load(libs, archives, callback);
        this.loadStatus = undefined;
        return pyodideInterface;
    }
    
    private async loadLibsArchives(...names: string[]) {
        let libs = [];
        for (let name of names) {
            libs.push(await firstValueFrom(this.assetsClient.get(name + ".whl", {responseType: "arraybuffer"})));
        }
        return libs;
    }
    
    private async getCode(name: ScriptName): Promise<string> {
        return firstValueFrom(this.assetsClient.get(`python/${name}.py`, {responseType: "text"}));
    }
}
