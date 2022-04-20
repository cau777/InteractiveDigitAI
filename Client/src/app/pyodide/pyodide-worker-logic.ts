import {IPyodide} from "./ipyodide";
import {IPyConsole} from "./ipy-console";
import {IPyProxy} from "./ipyproxy";
import {IPyodideOutputMessage} from "../pyodide-worker-messages";

export class PyodideWorkerLogic {
    private readonly pyodidePromise: Promise<IPyodide>;
    private pyconsole?: IPyConsole;
    
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
    
    public async select(code: string, data: any[]) {
        let namespace: IPyProxy | undefined = undefined;
        try {
            let pyodide = await this.pyodidePromise;
            
            if (this.pyconsole !== undefined) await this.close();
            
            namespace = pyodide.globals.get("dict")();
            
            await pyodide.loadPackagesFromImports(code);
            pyodide.runPython(code, {
                globals: namespace
            });
            
            let pyconsole: IPyConsole = namespace!.get("console");
            pyconsole.globals.update(namespace); // TODO: add 'data' to global namespace
            console.log(pyconsole.globals.toJs());
            
            pyconsole.stdout_callback = (o: string) => PyodideWorkerLogic.stdCallback(o, false);
            pyconsole.stderr_callback = (o: string) => PyodideWorkerLogic.stdCallback(o, true);
            
            this.pyconsole = pyconsole;
        } finally {
            namespace?.destroy();
        }
    }
    
    public async run(expression: string) {
        if (this.pyconsole === undefined)
            throw new Error("No script selected");
        
        const result = await this.pyconsole.push(expression);
        
        if ((await this.pyodidePromise).isPyProxy(result)) {
            let str = result.__str__();
            result.destroy();
            return str;
        }
        return result;
    }
    
    public async close() {
        if (this.pyconsole === undefined)
            throw new Error("No script selected");
        
        this.pyconsole.destroy();
    }
    
    private static stdCallback(content: string, isError: boolean) {
        let message: IPyodideOutputMessage = {
            action: "output",
            content: content,
            isError: isError
        };
        postMessage(message);
    }
}

declare function loadPyodide(): Promise<IPyodide>;
