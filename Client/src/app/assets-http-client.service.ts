import {Injectable} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";
import {environment} from "../environments/environment";

@Injectable({
    providedIn: 'root'
})
export class AssetsHttpClientService extends HttpClient {
    private baseUrl = environment.baseUrl;
    
    public override get(url: string, options: Object): Observable<any> {
        return super.get(this.baseUrl + "/assets/" + url, options);
    }
}
