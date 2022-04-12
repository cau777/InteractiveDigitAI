import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { TrainRoutingModule } from './train-routing.module';
import { TrainMainComponent } from './train-main/train-main.component';


@NgModule({
  declarations: [
    TrainMainComponent
  ],
  imports: [
    CommonModule,
    TrainRoutingModule
  ]
})
export class TrainModule { }
