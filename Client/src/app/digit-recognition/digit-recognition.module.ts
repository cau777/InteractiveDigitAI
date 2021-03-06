import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { DigitRecognitionRoutingModule } from './digit-recognition-routing.module';
import { DigitRecognitionMainComponent } from './digit-recognition-main/digit-recognition-main.component';
import {MatButtonModule} from "@angular/material/button";
import {MatIconModule} from "@angular/material/icon";
import {TranslateModule} from "@ngx-translate/core";


@NgModule({
  declarations: [
    DigitRecognitionMainComponent
  ],
    imports: [
        CommonModule,
        DigitRecognitionRoutingModule,
        MatButtonModule,
        MatIconModule,
        TranslateModule
    ]
})
export class DigitRecognitionModule { }
