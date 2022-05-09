import {Component} from '@angular/core';
import {PythonRunnerService} from "../../python-runner.service";
import {AiName, AiReposService, AiModel} from "../../ai-repos.service";
import {Log} from "../logs-view/logs-view.component";

type FormModel = (TrainModel) & {
    scriptName: AiName;
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
    public current: AiModel | null = null;
    public logs = new Map<string, Log>();
    public entries: [string, Log][] = [];
    public busy = false;
    public model: FormModel = {scriptName: "digit_recognition", action: "train", epochs: 0};
    private prevModels = new Map<AiName, AiModel>();
    // private loadedAis = new Map<AiName, AiModel>();
    
    public constructor(private pythonRunner: PythonRunnerService,
                       private aiRepos: AiReposService) {
        this.updateLogs = this.updateLogs.bind(this);
    }
    
    public async run() {
        if (this.busy) return;
        this.busy = true;
        
        let name = this.model.scriptName!;
        await this.pythonRunner.loadScript(name);
        
        let logName = (this.logs.size + 1) + ") " + this.model.action;
        this.logs.set(logName, {lines: []});
        
        let callback = (content: string, isError: boolean) => {
            let entry = this.logs.get(logName)!;
            entry.lines.push({content: content, isError: isError});
        };
        
        let interval = setInterval(this.updateLogs, 400);
        
        await this.aiRepos.loadFromServer(name, this.pythonRunner);
        if (!this.prevModels.has(name)) 
            this.prevModels.set(name, await this.pythonRunner.run("instance.save()"));
        
        if (this.model.action === "train") {
            console.log(await this.pythonRunner.run("instance.train()",{epochs: this.model.epochs}, callback));
            await this.saveAi(name);
        } else if (this.model.action === "test") {
            console.log(await this.pythonRunner.run("instance.test()", callback));
        }
        
        clearInterval(interval);
        this.updateLogs();
        this.busy = false;
    }
    
    private updateLogs() {
        this.entries = Array.from(this.logs.entries()).reverse();
    }
    
    private async saveAi(name: AiName) {
        let current: AiModel = await this.pythonRunner.run("instance.save()");
        await this.aiRepos.saveModelChanges(name, this.prevModels.get(name)!.params, current.params);
        this.prevModels.set(name, current);
    }
}
