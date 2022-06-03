import {Component} from '@angular/core';
import {PythonRunnerService} from "../../python-runner.service";
import {AiName, AiReposService, AiModel} from "../../ai-repos.service";
import {Log} from "../logs-view/logs-view.component";
import {arrayBufferToString, ObjDict} from "../../utils";
import {firstValueFrom} from "rxjs";
import {AssetsHttpClientService} from "../../assets-http-client.service";

type FormModel = (TrainModel) & {
    scriptName: AiName;
};

type TrainModel = {
    action: "train";
    epochs: number;
};

function formatObjDict(obj: ObjDict<any>) {
    return Object.entries(obj).map(([key, val]) => key + "=" + val).join(" ");
}

const trainSets = {
    digit_recognition: "mnist_train.dat"
};

const testSets = {
    digit_recognition: "mnist_test.dat"
}

@Component({
    selector: 'app-train-main',
    templateUrl: './train-main.component.html',
    styleUrls: ['./train-main.component.scss']
})
export class TrainMainComponent {
    public trainOptions: AiName[] = ["digit_recognition"]
    public logs = new Map<string, Log>();
    public entries: [string, Log][] = [];
    public busy = false;
    public model: FormModel = {scriptName: "digit_recognition", action: "train", epochs: 0};
    public result: string[] = [];
    private prevModels = new Map<AiName, AiModel>();
    
    private loadedTrainSet = false;
    private loadedTestSet = false;
    
    public constructor(private pythonRunner: PythonRunnerService,
                       private aiRepos: AiReposService,
                       private assetsClient: AssetsHttpClientService) {
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
        
        await this.aiRepos.loadModel(name, this.pythonRunner);
        if (!this.prevModels.has(name))
            this.prevModels.set(name, await this.pythonRunner.run("instance.save()"));
        
        if (this.model.action === "train") {
            await this.loadTrainSet(name);
            let result: ObjDict<any>[] = await this.pythonRunner.run("instance.train()", {epochs: this.model.epochs}, callback);
            this.result = result.map((val, index) => `Epoch ${index} with ${formatObjDict(val)}`);
            await this.saveAi(name);
        } else if (this.model.action === "test") {
            await this.loadTestSet(name);
            let result = await this.pythonRunner.run("instance.test()", {}, callback);
            this.result = [`Metrics: ${formatObjDict(result)}`];
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
        await this.aiRepos.saveModelChanges(name, this.prevModels.get(name)!, current);
        this.prevModels.set(name, current);
    }
    
    private async loadTrainSet(name: AiName) {
        if (this.loadedTrainSet) return;
        let data = await firstValueFrom(this.assetsClient.get(trainSets[name], {responseType: "arraybuffer"}));
        await this.pythonRunner.run("instance.load_train_set()", {data: arrayBufferToString(data)});
        this.loadedTrainSet = true;
    }
    
    private async loadTestSet(name: AiName) {
        if (this.loadedTestSet) return;
        let data = await firstValueFrom(this.assetsClient.get(testSets[name], {responseType: "arraybuffer"}));
        await this.pythonRunner.run("instance.load_test_set()", {data: arrayBufferToString(data)});
        this.loadedTestSet = true;
    }
}
