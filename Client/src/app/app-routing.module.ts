import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {NotFoundComponent} from "./not-found/not-found.component";

const routes: Routes = [
    {path: "", loadChildren: () => import("./home/home.module").then(o => o.HomeModule)},
    {
        path: "digit_recognition",
        loadChildren: () => import("./digit-recognition/digit-recognition.module").then(o => o.DigitRecognitionModule)
    },
    {path: "train", loadChildren: () => import("./train/train.module").then(o => o.TrainModule)},
    {path: "**", pathMatch: "full", component: NotFoundComponent}
];

@NgModule({
    imports: [RouterModule.forRoot(routes)],
    exports: [RouterModule]
})
export class AppRoutingModule {}
