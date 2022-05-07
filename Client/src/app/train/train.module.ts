import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';

import {TrainRoutingModule} from './train-routing.module';
import {TrainMainComponent} from './train-main/train-main.component';
import {MatSelectModule} from "@angular/material/select";
import {MatButtonModule} from "@angular/material/button";
import {FormsModule} from "@angular/forms";
import {MatInputModule} from "@angular/material/input";
import { LogsViewComponent } from './logs-view/logs-view.component';
import {MatExpansionModule} from "@angular/material/expansion";
import { LogDetailsComponent } from './log-details/log-details.component';
import { LogLinesDirective } from './log-lines.directive';


@NgModule({
    declarations: [
        TrainMainComponent,
        LogsViewComponent,
        LogDetailsComponent,
        LogLinesDirective
    ],
    imports: [
        CommonModule,
        TrainRoutingModule,
        MatSelectModule,
        MatButtonModule,
        FormsModule,
        MatInputModule,
        MatExpansionModule
    ]
})
export class TrainModule {}
