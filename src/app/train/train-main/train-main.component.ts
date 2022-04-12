import {Component, OnInit} from '@angular/core';
import {PythonRunnerService} from "../../python-runner.service";

@Component({
  selector: 'app-train-main',
  templateUrl: './train-main.component.html',
  styleUrls: ['./train-main.component.scss']
})
export class TrainMainComponent implements OnInit {
  public content = "";

  public constructor(private pythonRunner: PythonRunnerService) {
  }

  public async ngOnInit() {
    console.log("Result " + await this.pythonRunner.executeScript("test"));
  }
}
