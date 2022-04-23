import {Component} from '@angular/core';
import {PythonRunnerService} from "../../python-runner.service";
import {AiReposService, IAiModel} from "../../ai-repos.service";

@Component({
    selector: 'app-train-main',
    templateUrl: './train-main.component.html',
    styleUrls: ['./train-main.component.scss']
})
export class TrainMainComponent {
    public content = "";
    public current: IAiModel | null = null;
    
    public constructor(private pythonRunner: PythonRunnerService,
                       private aiRepos: AiReposService) {
    }
    
    public async run() {
        console.log("run")
        console.log("Result 1+1: " + await this.pythonRunner.run("1+1"));
        await this.pythonRunner.run("a = 1+1");
        console.log("Result 1+1: " + await this.pythonRunner.run("a"));
    
        console.log("a")
        
        // console.log("Result nn: " + JSON.stringify(await this.pythonRunner.run("instance.test()")));
        console.log("Result nn: " + JSON.stringify(await this.pythonRunner.run("instance.train(1)")));
        // console.log("Result nn: " + JSON.stringify(await this.pythonRunner.run("instance.test()")));
        // console.log("Result nn: " + JSON.stringify(await this.pythonRunner.run("instance.save()")));
    }
    
    public async getClick() {
        let result = await this.aiRepos.readAi("digit_recognition");
        this.current = result;
    }
    
    public async addOne() {
        if (this.current === null) return;
        await this.aiRepos.saveAiChanges("digit_recognition", this.current.params, this.current.params.map(o => o + 1));
    }
    
    public async load() {
        await this.pythonRunner.loadScript("digit_recognition");
        console.log("loaded")
    }
}
