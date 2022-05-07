import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {DigitRecognitionMainComponent} from "./digit-recognition-main/digit-recognition-main.component";

const routes: Routes = [{path: "", component: DigitRecognitionMainComponent}];

@NgModule({
    imports: [RouterModule.forChild(routes)],
    exports: [RouterModule]
})
export class DigitRecognitionRoutingModule {}
