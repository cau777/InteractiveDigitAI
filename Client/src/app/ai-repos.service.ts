import {Injectable} from '@angular/core';
import {AngularFireDatabase} from "@angular/fire/compat/database";
import {map, firstValueFrom} from "rxjs";
import {zip} from "./utils";

@Injectable({
    providedIn: 'root'
})
export class AiReposService {
    
    public constructor(private readonly db: AngularFireDatabase) {
    }
    
    public readAi(name: string) {
        const path = AiReposService.createModelPath(name);
        return firstValueFrom(this.db.object<IAiModel>(path)
            .snapshotChanges()
            .pipe(map(o => {
                    console.log(o.payload.exists())
                    return o.payload.exists() ? o.payload.val() : null
                })
            )
        );
    }
    
    public async createAi(name: string, model: IAiModel) {
        const path = AiReposService.createModelPath(name);
        await this.db.object<IAiModel>(path).query.ref
            .transaction(o => {
                return o ?? model;
            });
    }
    
    public async saveAiChanges(name: string, prevParams: number[], newParams: number[]) {
        const path = AiReposService.createModelPath(name);
        const paramsDelta = zip(newParams, prevParams).map(o => o[0] - o[1]);
        
        await this.db.object<IAiModel>(path).query.ref
            .transaction(o => {
                if (o === null) return o;
                
                const model: IAiModel = o;
                const newModel: IAiModel = {
                    version: model.version + 1,
                    params: zip(model.params, paramsDelta).map(o => o[0] + o[1])
                };
                console.log(JSON.stringify(newModel))
                return newModel;
            })
    }
    
    private static createModelPath(name: string) {
        return "/models/" + name;
    };
}

export interface IAiModel {
    version: number,
    params: number[]
}
