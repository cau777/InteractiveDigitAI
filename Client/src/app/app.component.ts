import {Component} from '@angular/core';
import {TranslateService} from "@ngx-translate/core";

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.scss']
})
export class AppComponent {
    public title = 'AIPlayground';
    
    public constructor(private translateService: TranslateService) {
        this.translateService.setDefaultLang("en");
        let lang = this.translateService.getBrowserLang() || "en";
        console.log(lang)
        // this.translateService.use(lang);
        this.translateService.use("pt")
    }
}
