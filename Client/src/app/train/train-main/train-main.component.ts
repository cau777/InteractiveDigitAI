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
    public current: IAiModel|null = null;
    
    public constructor(private pythonRunner: PythonRunnerService,
                       private aiRepos: AiReposService) {
    }
    
    public async test() {
        console.log("Result " + await this.pythonRunner.executeScript("test"));
    }
    
    public async getClick() {
        let result = await this.aiRepos.readAi("test");
        this.current = result;
    }
    
    public async addOne() {
        if (this.current === null) return;
        await this.aiRepos.saveAiChanges("test", this.current.params, this.current.params.map(o => o + 1));
    }
    
    public async create() {
        await this.aiRepos.createAi("test", {version: 0, params: Array.from(new Array(10).keys())});
    }
}
