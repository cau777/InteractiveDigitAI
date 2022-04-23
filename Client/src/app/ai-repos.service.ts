import {Injectable} from '@angular/core';
import {AngularFireDatabase} from "@angular/fire/compat/database";
import {map, firstValueFrom} from "rxjs";
import {zip} from "./utils";

export type AiName = "digit_recognition";

@Injectable({
    providedIn: 'root'
})
export class AiReposService {
    
    public constructor(private readonly db: AngularFireDatabase) {
    }
    
    public readAi(name: AiName) {
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
    
    public readAiJson(name: AiName) {
        const path = AiReposService.createModelPath(name);
        return firstValueFrom(this.db.object<IAiModel>(path)
            .snapshotChanges()
            .pipe(map(o => {
                    console.log(o.payload.exists())
                    return o.payload.exists() ? o.payload.toJSON() : null
                })
            )
        );
    }
    
    public async saveAiChanges(name: AiName, prevParams: number[] | null, newParams: number[]) {
        const path = AiReposService.createModelPath(name);
        
        await this.db.object<IAiModel>(path).query.ref
            .transaction(saveAiTransaction)
        
        function saveAiTransaction(model: IAiModel): IAiModel {
            if (model === null) {
                return {
                    version: 1,
                    params: newParams
                };
            }
            
            if (prevParams === null) return model;
            
            const paramsDelta = zip(newParams, prevParams).map(o => o[0] - o[1]);
            const newModel: IAiModel = {
                version: model.version + 1,
                params: zip(model.params, paramsDelta).map(o => o[0] + o[1])
            };
            console.log(JSON.stringify(newModel))
            return newModel;
        }
    }
    
    private static createModelPath(name: AiName) {
        return "/models/" + name;
    };
}

export interface IAiModel {
    version: number,
    params: number[]
}
