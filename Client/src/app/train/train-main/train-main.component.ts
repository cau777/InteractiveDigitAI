import {Component} from '@angular/core';
import {PythonRunnerService, ScriptName} from "../../python-runner.service";
import {AiName, AiReposService, IAiModel} from "../../ai-repos.service";
import {Log} from "../logs-view/logs-view.component";

type FormModel = (TrainModel) & {
    scriptName: ScriptName;
};

type TrainModel = {
    action: "train";
    epochs: number;
};

@Component({
    selector: 'app-train-main',
    templateUrl: './train-main.component.html',
    styleUrls: ['./train-main.component.scss']
})
export class TrainMainComponent {
    public trainOptions: AiName[] = ["digit_recognition"]
    public current: IAiModel | null = null;
    public logs = new Map<string, Log>();
    public entries: [string, Log][] = [];
    public busy = false;
    public model: FormModel = {scriptName:"digit_recognition",action: "train", epochs: 0};
    
    public constructor(private pythonRunner: PythonRunnerService,
                       private aiRepos: AiReposService) {
    }
    
    public async run() {
        if (this.busy) return;
        
        this.busy = true;
        
        console.log(this.model)
        let name = this.model.scriptName!;
        await this.pythonRunner.loadScript(name);
        
        let logName = (this.logs.size + 1) + ") " + this.model.action;
        this.logs.set(logName, {lines: []});
        
        let callback = (content: string, isError: boolean) => {
            let entry = this.logs.get(logName)!;
            entry.lines.push({content: content, isError: isError});
            this.entries = Array.from(this.logs.entries()).reverse();
        };
        
        if (this.model.action === "train") {
            // callback("sad999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999 9999999999999999999999999999999999999999999", false);
            // callback("asd", false);
            // callback("sfdad", false);
            // callback("sad", false);
            // callback("saxzcd", false);
            // callback("s===axz2cd", true);
            // callback("4412saxzcd", true);
            await this.pythonRunner.run(`instance.train(${this.model.epochs})`, callback);
        }
        
        this.busy = false;
    }
}
