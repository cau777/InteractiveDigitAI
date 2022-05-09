import {IPyodide} from "./ipyodide";
import {PyConsole} from "./py-console";
import {PyProxy} from "./pyproxy";
import {IPyodideOutputMessage} from "../pyodide-worker-messages";
import {ObjDict} from "../utils";

export class PyodideWorkerLogic {
    private readonly pyodidePromise: Promise<IPyodide>;
    private pyconsole?: PyConsole;
    
    public constructor(private baseUrl: string) {
        this.pyodidePromise = this.prepareEnv();
    }
    
    private async prepareEnv(): Promise<IPyodide> {
        let pyodide = await loadPyodide();
        
        await pyodide.loadPackage("micropip");
        
        const filename = "codebase-0.0.1-py3-none-any.whl";
        const codebaseUrl = this.baseUrl + ("/assets/python/" + filename);
        let list = await pyodide.runPythonAsync(`import micropip; await micropip.install('${codebaseUrl}'); str(micropip.list())`);
        console.log(list);
        
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
            console.log(pyconsole.globals.__str__());
            
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
        
        await this.clearParams(params);
        return result;
    }
    
    private async setParams(params: ObjDict<any>) {
        let pyodide = await this.pyodidePromise;
        let pyParams = pyodide.toPy(params);
        let instance = this.pyconsole.globals.get("instance").__dict__;
        instance.update(pyParams);
        pyParams.destroy();
        
        console.log(instance.keys().__str__());
    }
    
    private async clearParams(params: ObjDict<any>) {
        let pyodide = await this.pyodidePromise;
        let instance = this.pyconsole.globals.get("instance").__dict__;
        
        for (let key of Object.keys(params)) {
            let proxy = pyodide.toPy(key);
            instance.pop(proxy);
        }
    
        console.log(instance.keys().__str__());
    }
    
    public async close() {
        if (this.pyconsole === undefined)
            throw new Error("No script selected");
        
        this.pyconsole.destroy();
    }
    
    private static stdCallback(content: string, isError: boolean) {
        if (!content) return;
        let message: IPyodideOutputMessage = {
            action: "output",
            content: content,
            isError: isError
        };
        postMessage(message);
    }
}

declare function loadPyodide(): Promise<IPyodide>;
