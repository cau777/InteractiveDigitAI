import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';

import {TrainRoutingModule} from './train-routing.module';
import {TrainMainComponent} from './train-main/train-main.component';
import {MatSelectModule} from "@angular/material/select";


@NgModule({
    declarations: [
        TrainMainComponent
    ],
    imports: [
        CommonModule,
        TrainRoutingModule,
        MatSelectModule
    ]
})
export class TrainModule {}
