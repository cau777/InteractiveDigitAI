import {Component} from '@angular/core';
import {TranslateService} from "@ngx-translate/core";
import {environment} from "../environments/environment";

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.scss']
})
export class AppComponent {
    public constructor(private translateService: TranslateService) {
        this.translateService.setDefaultLang("en");
        let lang = environment.production ? (this.translateService.getBrowserLang() || "en") : "en";
        console.log("Selected language: ", lang)
        this.translateService.use(lang);
    }
}
