import {Injectable} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {firstValueFrom} from "rxjs";

declare interface IPyodide {
  runPython(code: string): any;

  loadPackagesFromImports(code: string): Promise<void>;
}

declare function loadPyodide(): Promise<IPyodide>;

@Injectable({
  providedIn: 'root'
})
export class PythonRunnerService {
  private readonly pyodide: Promise<IPyodide>;

  public constructor(private httpClient: HttpClient) {
    this.pyodide = this.prepareEnv();
  }

  private async prepareEnv() {
    let pyodide = await loadPyodide();

    return pyodide;
  }

  public async execute(code: string) {
    return (await this.pyodide).runPython(code);
  }

  public async executeScript(name: string) {
    let code = await this.getScript(name);
    let pyodide = await this.pyodide;
    await pyodide.loadPackagesFromImports(code);

    return pyodide.runPython(code);
  }

  private async getScript(name: string): Promise<string> {
    let observable = this.httpClient.get(`assets/python/${name}.py`, {responseType: "text"});
    return await firstValueFrom(observable);
  }
}
