import {Inject, Injectable} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {firstValueFrom} from "rxjs";
import {DOCUMENT, LocationStrategy} from "@angular/common";

declare interface IPyodide {
    /**
     * Runs a string of Python code from JavaScript, using pyodide.eval_code to evaluate the code. If the last statement
     * in the Python code is an expression (and the code doesn’t end with a semicolon), the value of the expression is returned.
     * @param code Python code to evaluate
     * @param options
     * @param options.globals An optional Python dictionary to use as the globals
     * @return The result of the Python code translated to JavaScript
     */
    runPython(code: string, options?: { globals: any }): any;
    
    /**
     * Run a Python code string with top level await using pyodide.eval_code_async to evaluate the code. 
     * Returns a promise which resolves when execution completes. 
     * If the last statement in the Python code is an expression (and the code doesn’t end with a semicolon), 
     * the returned promise will resolve to the value of this expression
     * @param code Python code to evaluate
     * @param options
     * @param options.globals An optional Python dictionary to use as the globals
     */
    runPythonAsync(code: string, options?: {globals: any}): Promise<any>;
    
    /**
     * Inspect a Python code chunk and use pyodide.loadPackage() to install any known packages that the code chunk imports
     * @param code The code to inspect
     * @param messageCallback The messageCallback argument of pyodide.loadPackage.
     * @param errorCallback The errorCallback argument of pyodide.loadPackage.
     */
    loadPackagesFromImports(code: string, messageCallback?: () => void, errorCallback?: () => void): Promise<void>;
    
    /**
     * Unpack an archive into a target directory
     * @param buffer The archive as an ArrayBuffer or TypedArray
     * @param format The format of the archive. Should be one of the formats recognized by shutil.unpack_archive.
     * @param options
     * @param options.extractDir The directory to unpack the archive into. Defaults to the working directory.
     */
    unpackArchive(buffer: ArrayBuffer, format?: string, options?: { extractDir: string }): void;
    
    /**
     * Load a package or a list of packages over the network. This installs the package in the virtual filesystem. 
     * The package needs to be imported from Python before it can be used.
     * @param names Either a single package name or URL or a list of them
     * @param messageCallback A callback, called with progress messages
     * @param errorCallback A callback, called with error/warning messages
     */
    loadPackage(names: string | string[], messageCallback?: (o:any) => void, errorCallback?: (o:any) => void): Promise<void>;
    
    /**
     * Imports a module and returns it.
     * @param mod_name The name of the module to import
     */
    pyimport(mod_name: string): any;
    
    /**
     * The list of packages that Pyodide has loaded
     */
    loadedPackages: {[key: string]: string};
}

declare function loadPyodide(): Promise<IPyodide>;


@Injectable({
    providedIn: 'root'
})
export class PythonRunnerService {
    private pyodidePromise?: Promise<IPyodide>;
    
    public constructor(private httpClient: HttpClient,
                       private location: LocationStrategy,
                       @Inject(DOCUMENT) private document: Document) {
    }
    
    private get env() : Promise<IPyodide> {
        this.pyodidePromise ??= this.prepareEnv();
        return this.pyodidePromise;
    }
    
    private async prepareEnv(): Promise<IPyodide> {
        let pyodide = await loadPyodide();
        
        await pyodide.loadPackage("micropip");
        
        const filename = "codebase-0.0.1-py3-none-any.whl";
        const codebaseUrl = this.document.location.origin + this.location.prepareExternalUrl("assets/python/" + filename);
        console.log(await pyodide.runPythonAsync(`import micropip; await micropip.install('${codebaseUrl}'); __import__('codebase'); str(micropip.list())`));
    
        console.log("Finished initializing pyodide")
        console.log("Installed packages" + Object.keys(pyodide.loadedPackages));
        
        return pyodide;
    }
    
    public async execute(code: string) {
        return (await this.env).runPython(code);
    }
    
    public async executeScript(name: string) {
        let code = await this.getScript(name);
        let pyodide = await this.env;
        await pyodide.loadPackagesFromImports(code);
        return pyodide.runPython(code);
    }
    
    private async getScript(name: string): Promise<string> {
        let observable = this.httpClient.get(`assets/python/${name}.py`, {responseType: "text"});
        return await firstValueFrom(observable);
    }
}
