import { TestBed } from '@angular/core/testing';

import { AssetsHttpClientService } from './assets-http-client.service';

describe('AssetsHttpClientService', () => {
  let service: AssetsHttpClientService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AssetsHttpClientService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
