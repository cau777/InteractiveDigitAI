import {ObjDict} from "../utils";
import {IPyodideRunner, PythonRunCallback} from "./IPyodideRunner";
import {PyConsole} from "./py-console";
import {IPyodide} from "./ipyodide";
import {PyProxy} from "./pyproxy";

declare function loadPyodide(): Promise<IPyodide>;

function assertInit(pyodide: IPyodide | undefined): asserts pyodide is IPyodide {
    if (pyodide === undefined) throw new ReferenceError("load() should be called first");
}

export class PyodideWorkerLogic implements IPyodideRunner {
    private pyodide?: IPyodide;
    private console?: PyConsole;
    
    public async load(libs: string[], libsArchives: ArrayBuffer[], output: PythonRunCallback) {
        let pyodide = await loadPyodide();
        
        await pyodide.loadPackage(libs, o => output(o.trim(), false), o => output(o.trim(), true));
        
        for (const lib of libsArchives)
            pyodide.unpackArchive(lib as any, "wheel");
        
        this.pyodide = pyodide;
    }
    
    public async select(code: string, params: ObjDict<any>) {
        assertInit(this.pyodide);
        let namespace: PyProxy | undefined = undefined;
        let instance: PyProxy | undefined = undefined;
        
        try {
            let pyodide = this.pyodide;
            if (this.console !== undefined)
                this.console.destroy();
            
            namespace = pyodide.globals.get("dict")();
            let pyParams = pyodide.toPy(params);
            namespace.update(pyParams);
            pyParams.destroy();
            
            await pyodide.loadPackagesFromImports(code);
            pyodide.runPython(code, {
                globals: namespace
            });
            
            instance = namespace!.get("instance");
            let console: PyConsole = instance.console.copy();
            console.globals.update(namespace);
            
            this.console = console;
        } finally {
            namespace?.destroy();
            instance?.destroy();
        }
    }
    
    public async run(expression: string, params: ObjDict<any>, callback: PythonRunCallback) {
        assertInit(this.pyodide);
        if (this.console === undefined)
            throw new Error("No script selected");
        this.setCallback(callback);
        
        await this.setParams(params);
        const result = await this.console.push(expression);
        
        let pyodide = await this.pyodide;
        if (pyodide.isPyProxy(result)) {
            let js = result.toJs({create_pyproxies: false, dict_converter: Object.fromEntries});
            result.destroy();
            return js;
        }
        
        await this.clearParams();
        return result;
    }
    
    private async setParams(params: ObjDict<any>) {
        let pyParams = this.pyodide!.toPy(params);
        let instance = this.console!.globals.get("instance");
        instance.params.update(pyParams);
        pyParams.destroy();
    }
    
    private async clearParams() {
        let instance = this.console!.globals.get("instance");
        instance.params.clear();
    }
    
    private setCallback(callback: PythonRunCallback | undefined) {
        if (callback === undefined) return;
        this.console!.stdout_callback = (o: string) => callback(o.trim(), false);
        this.console!.stderr_callback = (o: string) => callback(o.trim(), true);
    }
}

