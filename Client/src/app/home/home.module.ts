import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';

import {HomeRoutingModule} from './home-routing.module';
import {HomeMainComponent} from './home-main/home-main.component';
import {TranslateModule} from "@ngx-translate/core";

@NgModule({
    declarations: [
        HomeMainComponent
    ],
    imports: [
        CommonModule,
        HomeRoutingModule,
        TranslateModule
    ]
})
export class HomeModule {}
