import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {TrainMainComponent} from "./train-main/train-main.component";

const routes: Routes = [
  {path: "", component: TrainMainComponent}
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class TrainRoutingModule { }
