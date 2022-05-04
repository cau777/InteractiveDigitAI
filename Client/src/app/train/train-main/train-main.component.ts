import {Component} from '@angular/core';
import {PythonRunnerService} from "../../python-runner.service";
import {AiName, AiReposService, IAiModel} from "../../ai-repos.service";

@Component({
    selector: 'app-train-main',
    templateUrl: './train-main.component.html',
    styleUrls: ['./train-main.component.scss']
})
export class TrainMainComponent {
    public trainOptions: AiName[] = ["digit_recognition"]
    public current: IAiModel | null = null;
    
    public constructor(private pythonRunner: PythonRunnerService,
                       private aiRepos: AiReposService) {
    }
}
