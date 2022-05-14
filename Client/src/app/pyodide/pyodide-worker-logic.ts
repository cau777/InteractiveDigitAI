import {IPyodide} from "./ipyodide";
import {PyConsole} from "./py-console";
import {PyProxy} from "./pyproxy";
import {PyodideOutputMessage} from "../pyodide-worker-messages";
import {ObjDict} from "../utils";

export class PyodideWorkerLogic {
    private readonly pyodidePromise: Promise<IPyodide>;
    private pyconsole?: PyConsole;
    
    public constructor(private libs: string[],
                       private libsArchives: ArrayBuffer[]) {
        this.pyodidePromise = this.prepareEnv();
    }
    
    private async prepareEnv(): Promise<IPyodide> {
        let pyodide = await loadPyodide();
        
        await pyodide.loadPackage(this.libs);
        
        for (const lib of this.libsArchives)
            await pyodide.unpackArchive(lib, "wheel");
        
        return pyodide;
    }
    
    public async select(code: string, params: ObjDict<any>) {
        let namespace: PyProxy | undefined = undefined;
        let instance: PyProxy | undefined = undefined;
        
        try {
            let pyodide = await this.pyodidePromise;
            if (this.pyconsole !== undefined) await this.close();
            
            namespace = pyodide.globals.get("dict")();
            let pyParams = pyodide.toPy(params);
            namespace.update(pyParams);
            pyParams.destroy();
            
            await pyodide.loadPackagesFromImports(code);
            pyodide.runPython(code, {
                globals: namespace
            });
            
            instance = namespace!.get("instance");
            let pyconsole: PyConsole = instance.console.copy();
            pyconsole.globals.update(namespace);
            
            pyconsole.stdout_callback = (o: string) => PyodideWorkerLogic.stdCallback(o.trim(), false);
            pyconsole.stderr_callback = (o: string) => PyodideWorkerLogic.stdCallback(o.trim(), true);
            
            this.pyconsole = pyconsole;
        } finally {
            namespace?.destroy();
            instance?.destroy();
        }
    }
    
    public async run(expression: string, params: ObjDict<any>) {
        if (this.pyconsole === undefined)
            throw new Error("No script selected");
        
        await this.setParams(params);
        const result = await this.pyconsole.push(expression);
        
        let pyodide = await this.pyodidePromise;
        if (pyodide.isPyProxy(result)) {
            let js = result.toJs({create_pyproxies: false, dict_converter: Object.fromEntries});
            result.destroy();
            return js;
        }
        
        await this.clearParams();
        return result;
    }
    
    private async setParams(params: ObjDict<any>) {
        let pyodide = await this.pyodidePromise;
        let pyParams = pyodide.toPy(params);
        let instance = this.pyconsole.globals.get("instance");
        instance.params.update(pyParams);
        pyParams.destroy();
    }
    
    private async clearParams() {
        let instance = this.pyconsole.globals.get("instance");
        instance.params.clear();
    }
    
    public async close() {
        if (this.pyconsole === undefined)
            throw new Error("No script selected");
        
        this.pyconsole.destroy();
    }
    
    private static stdCallback(content: string, isError: boolean) {
        if (!content) return;
        let message: PyodideOutputMessage = {
            action: "output",
            content: content,
            isError: isError
        };
        postMessage(message);
    }
}

declare function loadPyodide(): Promise<IPyodide>;
