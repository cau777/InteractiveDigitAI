import {Component} from '@angular/core';
import {PythonRunnerService} from "../python-runner.service";

@Component({
    selector: 'app-header',
    templateUrl: './header.component.html',
    styleUrls: ['./header.component.scss']
})
export class HeaderComponent {
    public constructor(public pythonRunner: PythonRunnerService) {
    }
}
